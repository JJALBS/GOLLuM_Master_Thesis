"""
LLM featurizer components for GOLLuM (PEP 8 style).
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import init

from peft import LoraConfig, get_peft_model
from util.gollum_util import (
    average_pool,
    get_model_tokenizer_dim,
    get_target_layers,
    last_token_pool,
    weighted_average_pool,
)


class ProjectionLayer(nn.Module):
    """A single hidden-layer projector for embedding dimensionality reduction.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input embeddings.
    projection_dim : int, default=64
        Target dimensionality of the projected embeddings.
    device : torch.device, default=torch.device("cuda")
        Device to place module parameters on.
    data_type : torch.dtype, default=torch.float64
        Default floating-point dtype for module parameters.
    """

    def __init__(
        self,
        input_dim: int,
        projection_dim: int = 64,
        # Additional dynamic decision inputs
        device: torch.device = torch.device("cuda"),
        data_type: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(input_dim, projection_dim, device=device, dtype=data_type)

        # Xavier init and small positive bias for stability (behavior preserved)
        self.fc1.bias.data.fill_(0.01)
        init.xavier_uniform_(self.fc1.weight)

    def forward(self, x: Tensor) -> Tensor:
        """Project input embeddings and apply a nonlinearity.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape ``(N, input_dim)``.

        Returns
        -------
        Tensor
            Projected embeddings of shape ``(N, projection_dim)``.
        """
        x = self.fc1(x)
        x = F.elu(x)
        return x


class LLMFeaturizer(nn.Module):
    """LLM-based featurizer with optional LoRA and pooling strategies.

    This module wraps a Hugging Face transformer model and exposes a function
    to obtain fixed-size embeddings from tokenized input IDs and attention
    masks. Pooling choices include average, CLS token, last-token, and
    weighted-average pooling. An optional projection layer maps embeddings
    to a target dimensionality.

    Parameters
    ----------
    model_name : str, default="T5-base"
        Hugging Face model identifier.
    trainable : bool, default=True
        If ``True``, enables LoRA adapters for trainable fine-tuning; otherwise,
        parameters are frozen and inference runs under ``torch.no_grad()``.
    target_ratio : float, default=0.25
        Fraction of layers to target with LoRA (selected via :func:`get_target_layers`).
    from_top : bool, default=True
        If ``True``, choose target layers starting from the top of the model.
    lora_dropout : float, default=0.2
        Dropout probability inside LoRA adapters.
    modules_to_save : list[str] | None, default=["head"]
        Additional modules to keep in full precision with LoRA (forwarded to PEFT).
    pooling_method : str, default="cls"
        One of {"average", "cls", "last_token_pool", "weighted_average"}.
    normalize_embeddings : bool, default=False
        If ``True``, L2-normalize pooled embeddings.
    projection_dim : int | None, default=None
        If given, apply a :class:`ProjectionLayer` to reduce dimensionality.
    device : torch.device, default=torch.device("cuda")
        Device for the underlying model and projection layer.
    data_type : torch.dtype, default=torch.float64
        Default floating-point dtype for the model and projection layer.
    """

    def __init__(
        self,
        model_name: str = "T5-base",
        trainable: bool = True,
        target_ratio: float = 0.25,
        from_top: bool = True,
        lora_dropout: float = 0.2,
        modules_to_save: Optional[List[str]] = ["head"],
        pooling_method: str = "cls",
        normalize_embeddings: bool = False,
        projection_dim: Optional[int] = None,
        # Additional dynamic decision inputs
        device: torch.device = torch.device("cuda"),
        data_type: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        print(model_name, "for LLM")
        print(f"LLM using device: {device}")

        self.device = device
        self.data_type = data_type

        # Model, tokenizer, and hidden size from utility
        self.llm, self.tokenizer, self.llm_dim = get_model_tokenizer_dim(
            model_name, device
        )

        if trainable:
            target_modules = get_target_layers(self.llm, target_ratio, from_top)
            self.llm = get_peft_model(
                self.llm,
                LoraConfig(
                    r=4,
                    lora_alpha=16,
                    target_modules=target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                    use_rslora=True,
                    modules_to_save=modules_to_save,
                ),
            )
            self.llm.print_trainable_parameters()
        else:
            self.llm.requires_grad_(False)

        self.trainable = trainable
        self.embedding_dim = self.llm_dim
        self.pooling_method = pooling_method
        self.normalize_embeddings = normalize_embeddings
        self.input_dim = self.llm_dim

        if projection_dim is not None:
            self.projector: nn.Module = ProjectionLayer(
                input_dim=self.llm_dim,
                projection_dim=projection_dim,
                device=device,
                data_type=data_type,
            )
        else:
            self.projector = nn.Identity()

        self.llm = self.llm.to(device=device, dtype=data_type)
        self.projector = self.projector.to(device=device, dtype=data_type)

    def get_embeddings(self, x: Tensor, batch_size: int = 4) -> Tensor:
        """Compute pooled embeddings for a batch of tokenized inputs.

        The input contains ``input_ids`` and ``attention_mask`` concatenated
        along the last dimension: ``[input_ids, attention_mask]``. The function
        internally splits the last dimension into two equal parts.

        Parameters
        ----------
        x : Tensor
            Token tensor of shape ``(N, 2 * L)`` where the last dimension is
            the concatenation of ``input_ids`` and ``attention_mask``.
        batch_size : int, default=4
            Mini-batch size for forwarding through the LLM to control memory.

        Returns
        -------
        Tensor
            Pooled embeddings of shape ``(N, D)`` (``D`` is the model hidden size
            or the projected dimension if a projector is used).
        """

        x = x.to(dtype=self.data_type)
        self.llm = self.llm.to(dtype=self.data_type)

        n_points = x.size(0)
        ids_split = int(x.shape[-1] / 2)

        embedding_chunks = []

        current_idx = 0
        for start_idx in range(0, n_points, batch_size):
            end_idx = min(start_idx + batch_size, n_points)
            input_ids = x[start_idx:end_idx, :ids_split].long()
            attn_mask = x[start_idx:end_idx, ids_split:].long()

            if self.trainable:
                outputs = self.llm(input_ids=input_ids, attention_mask=attn_mask)
            else:
                self.llm.eval()
                with torch.no_grad():
                    outputs = self.llm(input_ids=input_ids, attention_mask=attn_mask)

            last_hidden_state = outputs.last_hidden_state

            if self.pooling_method == "average":
                pooled = average_pool(last_hidden_state, attn_mask)
            elif self.pooling_method == "cls":
                pooled = last_hidden_state[:, 0]
            elif self.pooling_method == "last_token_pool":
                pooled = last_token_pool(last_hidden_state, attn_mask)
            elif self.pooling_method == "weighted_average":
                pooled = weighted_average_pool(last_hidden_state, attn_mask)
            else:
                raise ValueError(f"Unknown pooling method: {self.pooling_method}")

            if self.normalize_embeddings:
                pooled = F.normalize(pooled, p=2, dim=1)

            batch_size = pooled.size(0)  # preserve original behavior
            embedding_chunks.append(pooled)
            current_idx += batch_size
            del outputs, last_hidden_state, pooled

        embeddings = torch.cat(embedding_chunks, dim=0)
        return embeddings

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass to obtain embeddings (with optional projection).

        Special handling is included for a 3-D input case produced by some
        BoTorch acquisition functions, where candidates are appended to a copy
        of training data along the sequence dimension.

        Parameters
        ----------
        x : Tensor
            Either a 2-D tensor ``(N, 2L)`` or a 3-D tensor ``(q, N+1, 2L)``
            where the last dimension concatenates ``input_ids`` and
            ``attention_mask``.

        Returns
        -------
        Tensor
            Embeddings of shape ``(N, D)`` or ``(q, N+1, D)`` depending on input.
        """
        # Case because of BoTorch acquisition function
        if x.dim() == 3:
            n_candidates, n_train, _ = x.shape
            train_data = x[0, : n_train - 1, :]

            # TODO: Update when batching across multiple training sets is needed
            all_candidates = x[:, n_train - 1, :]
            with torch.no_grad():
                train_embeddings = self.get_embeddings(train_data)
                all_candidate_embeddings = self.get_embeddings(all_candidates)

            train_embeddings = train_embeddings.unsqueeze(0).expand(
                n_candidates, -1, -1
            )
            candidate_embeddings = all_candidate_embeddings.unsqueeze(1)
            embeddings = torch.cat([train_embeddings, candidate_embeddings], dim=1)

        elif x.dim() == 2:
            embeddings = self.get_embeddings(x)

        return self.projector(embeddings)

    @property
    def output_dim(self) -> int:
        """Output dimension of the embeddings as seen by downstream models.

        Notes
        -----
        This mirrors the original behavior: if a :class:`ProjectionLayer` is
        used, this property still returns ``self.embedding_dim`` (LLM hidden
        size) rather than the projected dimension.
        """
        return (
            self.projector[-1].out_features
            if isinstance(self.projector, nn.Sequential)
            else self.embedding_dim
        )
