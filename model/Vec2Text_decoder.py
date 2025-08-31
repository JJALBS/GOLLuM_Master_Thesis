"""
Vec2Text decoder components (PEP 8 styled).

This module contains:
- :class:`EmbToSeq` - projects a fixed-size latent embedding to a sequence of
  pseudo-token embeddings for encoder inputs.
- :class:`InitialHypothesisGenerator` - a lightweight seq2seq generator that
  maps latent embeddings to token sequences using a LoRA-tuned T5 backbone.
- :class:`Vec2Text` - a two-stage "Base + Corrector" pipeline. The Base model
  proposes an initial sequence; the Corrector refines it using additional
  pseudo-tokens derived from (true, diff, pred) embeddings.
"""

from os import cpu_count
from typing import Any, List, Optional, Tuple
from itertools import chain

import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from transformers import T5ForConditionalGeneration

from model.gollum_LLM import LLMFeaturizer
from util.gollum_util import get_target_layers


def _filter_qv_only(model: nn.Module, candidates: list[str]) -> list[str]:
    """
    Keep only attention projection linears 'q' and 'v'.

    Parameters
    ----------
    model : nn.Module
        The T5 model whose modules are being adapted by LoRA.
    candidates : list[str]
        Candidate module names (from `get_target_layers`).

    Returns
    -------
    list[str]
        Module names restricted to {*.q, *.v}. If the intersection is empty
        (e.g., the chosen proportion excluded them), we fall back to *all*
        q/v modules found in the model.
    """
    # All q/v linear modules present in the model
    qv_all = {
        name
        for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear) and name.rsplit(".", 1)[-1] in {"q", "v"}
    }

    # Intersection with the current candidates
    filtered = [name for name in candidates if name in qv_all]

    # Fallback: if proportion filtering missed q/v entirely, take all q/v
    if not filtered:
        filtered = sorted(qv_all)

    print(f"[LoRA] Using {len(filtered)} q/v modules. Example(s): {filtered[:6]}")
    return filtered

class EmbToSeq(nn.Module):
    """Project a flat embedding into a sequence of pseudo-token vectors.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input embedding (``d``).
    hidden_dim : int
        Hidden size for the internal MLP (``d'``).
    output_dim : tuple[int, int]
        Target output shape (``seq_len``, ``token_dim``) = (``s``, ``d_enc``).
    dropout : float
        Dropout probability applied between layers.

    Notes
    -----
    The output is reshaped to ``(B, seq_len, token_dim)``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: Tuple[int, int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim

        seq_len, token_dim = output_dim
        flat_output_dim = seq_len * token_dim

        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_dim, flat_output_dim),
        )

    def forward(self, emb: Tensor) -> Tensor:
        """Transform a batch of embeddings to a pseudo-token sequence.

        Parameters
        ----------
        emb : Tensor
            Input tensor of shape ``(B, input_dim)``.

        Returns
        -------
        Tensor
            Output tensor of shape ``(B, seq_len, token_dim)``.
        """
        seq = self.layer(emb)
        seq = seq.view(emb.size(0), *self.output_dim)
        return seq


class InitialHypothesisGenerator(nn.Module):
    """Seq2seq base model that generates an initial token hypothesis.

    The model takes a latent embedding, projects it to encoder embeddings with
    :class:`EmbToSeq`, and decodes a token sequence using a LoRA-tuned
    :class:`~transformers.T5ForConditionalGeneration` backbone.

    Parameters
    ----------
    latent_dim : int
        Input latent embedding dimension.
    EmbToSeq_hidden_dim : int
        Hidden size for the :class:`EmbToSeq` MLP.
    EmbToSeq_dropout : float
        Dropout probability for :class:`EmbToSeq`.
    num_repeat_tokens : int, default=8
        Number of pseudo-token embeddings to feed as encoder inputs.
    LoRA_target_proportion : float, default=0.25
        Proportion of target layers for LoRA adaptation (from top).
    LoRA_r : int, default=1
        LoRA rank hyperparameter.
    LoRA_alpha : int, default=16
        LoRA alpha hyperparameter.
    LoRA_dropout : float, default=0.2
        Dropout probability within LoRA adapters.
    pad_idx_data : int, default=0
        Integer ID of the pad token (SELFIES [nop]).
    vocab_size : int, default=0
        Vocabulary size; used to resize token embeddings.
    eos_idx_data : int | None, default=None
        EOS token ID for the decoder. If ``None``, it defaults to ``pad_idx_data``.
    bos_idx_data : int | None, default=None
        Optional BOS token ID; if ``None``, ``pad_idx_data`` is used.
    device : torch.device, default=torch.device("cuda")
        Compute device.
    data_type : torch.dtype, default=torch.float32
        Floating-point dtype for module parameters and inputs.
    """

    def __init__(
        self,
        # EmbToSeq input arguments
        latent_dim: int,
        EmbToSeq_hidden_dim: int,
        EmbToSeq_dropout: float,
        # LLM input arguments
        num_repeat_tokens: int = 8,
        LoRA_target_proportion: float = 0.25,
        LoRA_r: int = 1,
        LoRA_alpha: int = 16,
        LoRA_dropout: float = 0.2,
        # Additional inputs for token resizing
        pad_idx_data: int = 0,
        vocab_size: int = 0,
        eos_idx_data: Optional[int] = None,
        bos_idx_data: Optional[int] = None,
        # Additional inputs
        device: torch.device = torch.device("cuda"),
        data_type: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.device = device
        self.data_type = data_type
        self.num_repeat_tokens = num_repeat_tokens

        # 1) Seq2Seq backbone (LoRA-tuned T5)
        base_t5 = T5ForConditionalGeneration.from_pretrained("t5-small")
        for name, module in base_t5.named_modules():
            # T5 uses DenseReluDense or DenseGatedActDense for FFN
            if name.endswith(("DenseReluDense", "DenseGatedActDense")):
                for param in module.parameters():
                    param.requires_grad = False
        target_layers = _filter_qv_only(
            base_t5,
            get_target_layers(base_t5, proportion=LoRA_target_proportion, from_top=True),
        )
        self.enc_dec = get_peft_model(
            base_t5,
            LoraConfig(
                r=LoRA_r,
                lora_alpha=LoRA_alpha,
                target_modules=target_layers,
                lora_dropout=LoRA_dropout,
                bias="none",
                use_rslora=True,
            ),
        ).to(self.device, self.data_type)

        hidden_dim = base_t5.get_input_embeddings().embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        assert vocab_size > 0, "vocab_size must be > 0 to resize token embeddings."
        if eos_idx_data is None:
            eos_idx_data = pad_idx_data

        self.enc_dec.resize_token_embeddings(vocab_size)
        self.enc_dec.config.pad_token_id = pad_idx_data
        self.enc_dec.config.eos_token_id = eos_idx_data
        self.enc_dec.config.decoder_start_token_id = (
            bos_idx_data if bos_idx_data is not None else pad_idx_data
        )
        self.pad_token_id = pad_idx_data

        out_shape = (self.num_repeat_tokens, self.hidden_dim)
        self.emb_to_seq = EmbToSeq(
            input_dim=latent_dim,
            hidden_dim=EmbToSeq_hidden_dim,
            output_dim=out_shape,
            dropout=EmbToSeq_dropout,
        ).to(self.device, self.data_type)

    def forward(self, emb: Tensor, labels: Optional[Tensor] = None) -> Any:
        """Run the base seq2seq given a batch of latent embeddings.

        Parameters
        ----------
        emb : Tensor
            Latent embeddings of shape ``(B, latent_dim)``.
        labels : Tensor | None, default=None
            Optional target token IDs of shape ``(B, L)`` used for
            teacher-forced training (passed to the T5 loss).

        Returns
        -------
        Any
            Hugging Face model output (includes ``logits`` and optionally ``loss``).
        """
        emb = emb.to(self.device, dtype=self.data_type)
        enc_embeds = self.emb_to_seq(emb)  # (B, S, H)
        attn = torch.ones(
            enc_embeds.size()[:2], device=self.device, dtype=torch.long
        )  # (B, S)
        return self.enc_dec(
            inputs_embeds=enc_embeds, attention_mask=attn, labels=labels
        )

    @torch.no_grad()
    def generate(self, emb: Tensor, max_length: int) -> torch.LongTensor:
        """Autoregressively generate token IDs from a latent embedding batch.

        Parameters
        ----------
        emb : Tensor
            Latent embeddings of shape ``(B, latent_dim)``.
        max_length : int
            Maximum generated length for the decoder.

        Returns
        -------
        torch.LongTensor
            Generated token IDs of shape ``(B, L_gen)``.
        """
        emb = emb.to(self.device, dtype=self.data_type)
        enc_embeds = self.emb_to_seq(emb)  # (B, S, H)
        attn = torch.ones(enc_embeds.size()[:2], device=self.device, dtype=torch.long)

        gen_kwargs = dict(
            max_length=max_length,
            num_beams=1,
            no_repeat_ngram_size=0,
            repetition_penalty=1.0,
        )
        return self.enc_dec.generate(
            inputs_embeds=enc_embeds, attention_mask=attn, **gen_kwargs
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        epochs: int = 10,
        use_amp: bool = True,
        # Hyperparameter tuning pruner setting
        trial: Optional[Any] = None,
        report_offset: int = 0,
        pad_idx_data: int = 0,
        weight: Optional[Tensor] = None,
        es_patience: int = 8,     
        es_min_delta: float = 1e-4,  
        train_eval: bool = True,
    ) -> float:
        """Train the base model with cross-entropy over non-pad tokens.

        Parameters
        ----------
        train_loader : DataLoader
            Yields ``(emb, labels)`` pairs.
        val_loader : DataLoader | None
            Optional validation loader.
        optimizer : torch.optim.Optimizer
            Optimizer for the base model parameters.
        epochs : int, default=10
            Number of training epochs.
        use_amp : bool, default=False
            Enable CUDA AMP mixed precision.
        trial : Any | None, default=None
            Optuna trial for pruning/reporting (optional).
        report_offset : int, default=0
            Optional offset for external reporting.
        pad_idx_data : int, default=0
            Pad ID to be ignored in the loss.
        weight : Tensor | None, default=None
            Class weights for cross-entropy.

        Returns
        -------
        float
            Best validation cross-entropy (if validation is provided) or the
            last training cross-entropy otherwise.
        """
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        best_val_ce: float = float("inf")
        last_train_ce: float = float("inf")

        no_improve: int = 0

        self.train_loss_per_epoch: List[float] = []
        self.val_loss_per_epoch: List[float] = []
        self.train_eval_loss_per_epoch: List[float] = []

        for epoch in range(epochs):
            self.train()
            total_loss = 0.0
            total_count = 0

            for emb_batch, labels in train_loader:
                emb_batch = emb_batch.to(self.device, dtype=self.data_type)
                labels = labels.to(self.device)

                pad_id = self.pad_token_id
                masked_labels = labels.clone()
                masked_labels[(labels == pad_id) | (labels == pad_idx_data)] = -100

                optimizer.zero_grad()
                if use_amp:
                    with torch.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                        outputs = self(emb_batch, labels=masked_labels)
                        logits = outputs.logits
                        preds = logits[:, :-1, :].float()
                        targets = masked_labels[:, 1:]
                        loss = F.cross_entropy(
                            preds.reshape(-1, self.vocab_size),
                            targets.reshape(-1),
                            ignore_index=-100,
                            reduction="mean",
                        )
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = self(emb_batch, labels=masked_labels)
                    logits = outputs.logits
                    preds = logits[:, :-1, :].float()
                    targets = masked_labels[:, 1:]
                    loss = F.cross_entropy(
                        preds.reshape(-1, self.vocab_size),
                        targets.reshape(-1),
                        ignore_index=-100,
                        reduction="mean",
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    optimizer.step()

                nonpad = (targets != -100).sum().item()
                total_loss += loss.item() * max(nonpad, 1)
                total_count += nonpad

            last_train_ce = total_loss / max(1, total_count)
            self.train_loss_per_epoch.append(last_train_ce)

            avg_val_ce: Optional[float] = None
            if val_loader is not None:
                self.eval()
                val_loss = 0.0
                val_count = 0
                train_val_loss = 0
                train_val_count = 0

                with torch.no_grad():
                    for emb_batch, labels in val_loader:
                        emb_batch = emb_batch.to(self.device, dtype=self.data_type)
                        labels = labels.to(self.device)

                        pad_id = self.pad_token_id
                        masked_labels = labels.clone()
                        masked_labels[(labels == pad_id) | (labels == pad_idx_data)] = -100

                        outputs = self(emb_batch, labels=masked_labels)
                        logits = outputs.logits
                        preds = logits[:, :-1, :].float()
                        targets = masked_labels[:, 1:]
                        loss = F.cross_entropy(
                            preds.reshape(-1, self.vocab_size),
                            targets.reshape(-1),
                            ignore_index=-100,
                            reduction="mean",
                        )

                        nonpad = (targets != -100).sum().item()
                        val_loss += loss.item() * max(nonpad, 1)
                        val_count += nonpad

                    if train_eval:
                        for emb_batch, labels in train_loader:
                            emb_batch = emb_batch.to(self.device, dtype=self.data_type)
                            labels = labels.to(self.device)

                            pad_id = self.pad_token_id
                            masked_labels = labels.clone()
                            masked_labels[(labels == pad_id) | (labels == pad_idx_data)] = -100

                            outputs = self(emb_batch, labels=masked_labels)
                            logits = outputs.logits
                            preds = logits[:, :-1, :].float()
                            targets = masked_labels[:, 1:]
                            loss = F.cross_entropy(
                                preds.reshape(-1, self.vocab_size),
                                targets.reshape(-1),
                                ignore_index=-100,
                                reduction="mean",
                            )

                            nonpad = (targets != -100).sum().item()
                            train_val_loss += loss.item() * max(nonpad, 1)
                            train_val_count += nonpad
                        avg_train_val_ce = train_val_loss / max(1, train_val_count)
                        self.train_eval_loss_per_epoch.append(avg_train_val_ce)

                avg_val_ce = val_loss / max(1, val_count)
                self.val_loss_per_epoch.append(avg_val_ce)

                # Optuna pruning
                if trial is not None:
                    step = report_offset + epoch
                    trial.report(float(avg_val_ce if avg_val_ce is not None else last_train_ce), step=step)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                # ---- Early stopping (Base) ----
                if avg_val_ce is not None:
                    if avg_val_ce < best_val_ce - es_min_delta:
                        best_val_ce = avg_val_ce
                        no_improve = 0
                    else:
                        no_improve += 1
                        if no_improve >= es_patience:
                            break

            if avg_val_ce is not None:
                print(
                    f"[Initial Hypothesis] Epoch {epoch + 1}/{epochs} — Train CE: {last_train_ce:.4f} — "
                    f"Val CE: {avg_val_ce:.4f}"
                )
            else:
                print(f"[Initial Hypothesis] Epoch {epoch + 1}/{epochs} — Train CE: {last_train_ce:.4f}")
        return float(best_val_ce if val_loader is not None else last_train_ce)


class Vec2Text(nn.Module):
    """Two-stage decoder: Base hypothesis + Corrector refinement.

    The Base model proposes initial token IDs from a latent embedding. The
    Corrector then refines this sequence by concatenating pseudo-tokens built
    from the true embedding, the difference (true - guess), and the guess
    embedding itself with the token embeddings of the current hypothesis.

    Parameters
    ----------
    latent_dim : int
        Input latent embedding dimension.
    EmbToSeq_hidden_dim : int
        Hidden size for the :class:`EmbToSeq` MLPs.
    EmbToSeq_dropout : float
        Dropout probability for :class:`EmbToSeq` MLPs.
    num_repeat_tokens : int
        Number of pseudo-token embeddings per head (real/diff/pred).
    LoRA_target_proportion : float, default=0.25
        Proportion of layers targeted for LoRA (from top).
    LoRA_r : int, default=1
        LoRA rank.
    LoRA_alpha : int, default=16
        LoRA alpha.
    LoRA_dropout : float, default=0.2
        LoRA dropout probability.
    model_name : str, default="T5-base"
        Hugging Face model for the :class:`LLMFeaturizer` embedder.
    trainable : bool, default=True
        Whether the embedder uses trainable LoRA adapters.
    target_ratio : float, default=0.25
        Ratio for target layer selection in the embedder.
    from_top : bool, default=True
        If ``True``, select target layers starting from the top.
    lora_dropout : float, default=0.2
        Dropout inside the embedder's LoRA adapters.
    modules_to_save : list[str] | None, default=["head"]
        Modules to save in full precision with PEFT.
    pooling_method : str, default="cls"
        Pooling method used by the embedder ("cls", "average", etc.).
    normalize_embeddings : bool, default=False
        Whether to L2-normalize embedder outputs.
    projection_dim : int | None, default=None
        Optional projection dimension for the embedder outputs.
    LLMFeaturizer_LoRA_ckpt : dict | None, default=None
    LLMFeaturizer_projector_ckpt : dict | None, default=None
        Optional checkpoints loaded into the embedder.
    pad_idx_data : int, default=0
        Integer ID of the pad token (SELFIES [nop]).
    vocab_size : int, default=0
        Vocabulary size for resizing token embeddings in the Corrector.
    eos_idx_data : int | None, default=None
        EOS ID for the Corrector; defaults to ``pad_idx_data`` when ``None``.
    bos_idx_data : int | None, default=None
        BOS ID for the Corrector; defaults to ``pad_idx_data`` when ``None``.
    max_length : int, default=128
        Maximum decoding length for hypotheses.
    device : torch.device, default=torch.device("cuda")
        Compute device.
    data_type : torch.dtype, default=torch.float32
        Floating-point dtype for module parameters and inputs.
    """

    def __init__(
        self,
        # EmbToSeq input arguments
        latent_dim: int,
        EmbToSeq_hidden_dim: int,
        EmbToSeq_dropout: float,
        # LLM input arguments
        num_repeat_tokens: int,
        LoRA_target_proportion: float = 0.25,
        LoRA_r: int = 1,
        LoRA_alpha: int = 16,
        LoRA_dropout: float = 0.2,
        # LLMFeaturizer input arguments
        model_name: str = "T5-base",
        trainable: bool = True,
        target_ratio: float = 0.25,
        from_top: bool = True,
        lora_dropout: float = 0.2,
        modules_to_save: Optional[List[str]] = ["head"],
        pooling_method: str = "cls",
        normalize_embeddings: bool = False,
        projection_dim: Optional[int] = None,
        LLMFeaturizer_LoRA_ckpt: Optional[dict] = None,
        LLMFeaturizer_projector_ckpt: Optional[dict] = None,
        # Additional inputs for token resizing
        pad_idx_data: int = 0,
        vocab_size: int = 0,
        eos_idx_data: Optional[int] = None,
        bos_idx_data: Optional[int] = None,
        # Corrector input arguments
        max_length: int = 128,
        # Additional inputs
        device: torch.device = torch.device("cuda"),
        data_type: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.max_length = max_length
        self.device = device
        self.data_type = data_type
        self.num_repeat_tokens = num_repeat_tokens

        # --- Base (Initial Hypothesis) ---
        self.initial_hypothesis = InitialHypothesisGenerator(
            latent_dim=latent_dim,
            EmbToSeq_hidden_dim=EmbToSeq_hidden_dim,
            EmbToSeq_dropout=EmbToSeq_dropout,
            LoRA_target_proportion=LoRA_target_proportion,
            LoRA_r=LoRA_r,
            LoRA_alpha=LoRA_alpha,
            LoRA_dropout=LoRA_dropout,
            pad_idx_data=pad_idx_data,
            vocab_size=vocab_size,
            eos_idx_data=eos_idx_data,
            bos_idx_data=bos_idx_data,
            device=device,
            data_type=data_type,
        )

        # --- Corrector (LoRA-tuned T5) ---
        base_t5 = T5ForConditionalGeneration.from_pretrained("t5-small")
        # Freeze Feed-Forward (FFN) modules only
        for name, module in base_t5.named_modules():
            # T5 uses DenseReluDense or DenseGatedActDense for FFN
            if name.endswith(("DenseReluDense", "DenseGatedActDense")):
                for param in module.parameters():
                    param.requires_grad = False
        target_layers = _filter_qv_only(
            base_t5,
            get_target_layers(base_t5, proportion=LoRA_target_proportion, from_top=True),
        )
        self.corrector = get_peft_model(
            base_t5,
            LoraConfig(
                r=LoRA_r,
                lora_alpha=LoRA_alpha,
                target_modules=target_layers,
                lora_dropout=LoRA_dropout,
                bias="none",
                use_rslora=True,
            ),
        ).to(self.device, self.data_type)

        hidden_dim = base_t5.get_input_embeddings().embedding_dim
        self.hidden_dim = hidden_dim
        self.pad_token_id = self.corrector.config.pad_token_id
        self.vocab_size = vocab_size

        assert vocab_size > 0, "vocab_size must be > 0 to resize token embeddings."
        if eos_idx_data is None:
            eos_idx_data = pad_idx_data

        self.corrector.resize_token_embeddings(vocab_size)
        self.corrector.config.pad_token_id = pad_idx_data
        self.corrector.config.eos_token_id = eos_idx_data
        self.corrector.config.decoder_start_token_id = (
            bos_idx_data if bos_idx_data is not None else pad_idx_data
        )
        self.pad_token_id = pad_idx_data

        # --- Corrector's EmbToSeq preparation ---
        out_shape = (self.num_repeat_tokens, self.hidden_dim)
        self.real_Emb2Seq = EmbToSeq(
            latent_dim, EmbToSeq_hidden_dim, out_shape, EmbToSeq_dropout
        ).to(self.device, self.data_type)
        self.difference_Emb2Seq = EmbToSeq(
            latent_dim, EmbToSeq_hidden_dim, out_shape, EmbToSeq_dropout
        ).to(self.device, self.data_type)
        self.predict_Emb2Seq = EmbToSeq(
            latent_dim, EmbToSeq_hidden_dim, out_shape, EmbToSeq_dropout
        ).to(self.device, self.data_type)

        assert (
            self.initial_hypothesis.enc_dec.get_input_embeddings().embedding_dim
            == self.hidden_dim
        ), "BaseModel and corrector hidden dims differ!"

        # --- Embedder (LLMFeaturizer) ---
        embedder = LLMFeaturizer(
            model_name=model_name,
            trainable=trainable,
            target_ratio=target_ratio,
            from_top=from_top,
            lora_dropout=lora_dropout,
            modules_to_save=modules_to_save,
            pooling_method=pooling_method,
            normalize_embeddings=normalize_embeddings,
            projection_dim=projection_dim,
            device=device,
            data_type=data_type,
        )
        embedder.projector.load_state_dict(LLMFeaturizer_projector_ckpt, strict=True)
        embedder.llm.load_state_dict(LLMFeaturizer_LoRA_ckpt, strict=False)

        embedder.projector.to(self.device, self.data_type)
        embedder.llm.to(self.device)

        self.embedder = embedder.eval()

    def forward(
        self,
        max_loop: int,
        latent_emb: torch.FloatTensor,
        true_emb: torch.FloatTensor,
        early_stop_tol: float = 1e-4,
        patience: int = 5,
    ) -> torch.LongTensor:
        """Iteratively refine a hypothesis sequence with the Corrector.

        Parameters
        ----------
        max_loop : int
            Maximum number of refinement iterations.
        latent_emb : torch.FloatTensor
            Latent embeddings that condition the Base model.
        true_emb : torch.FloatTensor
            Target embeddings used for similarity scoring and pseudo-tokens.
        early_stop_tol : float, default=1e-4
            Minimum improvement in cosine similarity to reset patience.
        patience : int, default=5
            Early-stop after ``patience`` iterations without sufficient
            improvement.

        Returns
        -------
        torch.LongTensor
            Best token IDs found during refinement (shape ``(B, L)``).
        """
        self.initial_hypothesis.eval()
        self.corrector.eval()
        self.embedder.eval()

        with torch.no_grad():
            enc_tokens = self.initial_hypothesis.emb_to_seq(
                latent_emb.to(self.device, self.data_type)
            )
            attn = torch.ones(
                (enc_tokens.size(0), enc_tokens.size(1)),
                device=self.device,
                dtype=torch.long,
            )
            gen_kwargs = dict(
                max_length=self.max_length,
                num_beams=1,
                no_repeat_ngram_size=0,
                repetition_penalty=1.0,
            )
            current_ids = self.initial_hypothesis.enc_dec.generate(
                inputs_embeds=enc_tokens, attention_mask=attn, **gen_kwargs
            )
            best_ids = current_ids
            best_score = -float("inf")
            bad_rounds = 0

            for _ in range(max_loop):
                pad = self.pad_token_id
                attention = (current_ids != pad).long()  # (B, L)
                x_for_embedder = torch.cat([current_ids, attention], dim=1)  # (B, 2L)

                guess_emb = self.embedder(x_for_embedder)
                diff_emb = true_emb - guess_emb

                real_seq = self.real_Emb2Seq(true_emb)
                diff_seq = self.difference_Emb2Seq(diff_emb)
                pred_seq = self.predict_Emb2Seq(guess_emb)

                token_embeds = self.corrector.get_input_embeddings()(current_ids).to(
                    self.data_type
                )

                inputs_embeds = torch.cat(
                    [real_seq, diff_seq, pred_seq, token_embeds], dim=1
                )
                s = self.num_repeat_tokens
                pseudo_mask = torch.ones(
                    (inputs_embeds.size(0), 3 * s),
                    device=self.device,
                    dtype=torch.long,
                )
                token_mask = (current_ids != self.pad_token_id).long()
                attention_mask = torch.cat([pseudo_mask, token_mask], dim=1)

                next_ids = self.corrector.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )

                nxt_attn = (next_ids != self.pad_token_id).long()
                new_emb = self.embedder(torch.cat([next_ids, nxt_attn], dim=1))

                score = F.cosine_similarity(new_emb, true_emb, dim=1).mean().item()
                if score - best_score >= early_stop_tol:
                    best_score = score
                    best_ids = next_ids
                    bad_rounds = 0
                else:
                    bad_rounds += 1
                    if bad_rounds >= patience:
                        break

                current_ids = next_ids

            return best_ids

    @staticmethod
    def _pad_to_len(ids: torch.LongTensor, target_len: int, pad_id: int) -> torch.LongTensor:
        """Right-pad or truncate a ``(B, L)`` tensor to ``(B, target_len)``.

        Parameters
        ----------
        ids : torch.LongTensor
            Input IDs of shape ``(B, L)``.
        target_len : int
            Desired output length.
        pad_id : int
            Pad token ID used for right-padding.

        Returns
        -------
        torch.LongTensor
            Padded or truncated IDs of shape ``(B, target_len)``.
        """
        bsz, length = ids.shape
        if length == target_len:
            return ids
        if length < target_len:
            pad = ids.new_full((bsz, target_len - length), pad_id)
            return torch.cat([ids, pad], dim=1)
        return ids[:, :target_len]

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        base_optimizer: torch.optim.Optimizer,
        corr_optimizer: torch.optim.Optimizer,
        base_epochs: int = 100,
        corr_epochs: int = 100,
        corr_batch_size: int = 32,
        trial: Optional[Any] = None,
        pad_idx_data: int = 0,
        weight: Optional[Tensor] = None,
        report_offset: int = 0,
        es_patience_base: int = 8,      # <-- add
        es_patience_corr: int = 8,      # <-- add
        es_min_delta: float = 1e-4,     # <-- add
        train_eval: bool = True,
    ) -> float:
        """Train the Base model, build a Corrector dataset, then train Corrector.

        The loss for both stages is cross-entropy over non-pad targets. The
        Corrector is trained on tuples constructed from the Base outputs and
        the embedder: (true emb, hypothesis emb, initial IDs, ground truth IDs).

        Parameters
        ----------
        train_loader : DataLoader
            Loader yielding ``(latent_emb, labels)`` pairs used by the Base.
        val_loader : DataLoader | None
            Optional validation loader for the Base and Corrector.
        base_optimizer : torch.optim.Optimizer
            Optimizer for the Base model.
        corr_optimizer : torch.optim.Optimizer
            Optimizer for the Corrector.
        base_epochs : int, default=100
            Number of training epochs for the Base model.
        corr_epochs : int, default=100
            Number of epochs for the Corrector.
        corr_batch_size : int, default=32
            Batch size for the Corrector training loader.
        trial : Any | None, default=None
            Optional Optuna trial for pruning/reporting.
        pad_idx_data : int, default=0
            Pad ID to be ignored in the loss.
        weight : Tensor | None, default=None
            Class weights for cross-entropy.
        report_offset : int, default=0
            Optional step offset for external reporting.

        Returns
        -------
        float
            Best (lowest) validation CE if a validation set is provided;
            otherwise, the final training CE of the Corrector stage.
        """
        # ---- Phase 1: Train initial-hypothesis (Base Model) ----
        self.initial_hypothesis.fit(  # noqa: F841
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=base_optimizer,
            epochs=base_epochs,
            use_amp=True,
            trial=trial,
            report_offset=report_offset,
            pad_idx_data=pad_idx_data,
            weight=weight,
            es_patience=es_patience_base,     # <-- add
            es_min_delta=es_min_delta,        # <-- add
            train_eval=train_eval
        )

        # Freeze Base Model & embedder
        self.initial_hypothesis.eval()
        self.embedder.eval()
        for p in self.initial_hypothesis.parameters():
            p.requires_grad_(False)
        for p in self.embedder.parameters():
            p.requires_grad_(False)

        # ---- Phase 2: Build Corrector dataset ----
        corrector_data = []
        for emb_batch, true_labels in train_loader:
            emb_batch = emb_batch.to(self.device, dtype=self.data_type)

            # 2.1 Generate initial hypothesis IDs x^(0)
            with torch.no_grad():
                enc_tokens = self.initial_hypothesis.emb_to_seq(emb_batch)
                s = enc_tokens.size(1)
                init_ids = self.initial_hypothesis.generate(
                    emb_batch, max_length=self.max_length
                )
                init_ids = self._pad_to_len(init_ids, self.max_length, self.pad_token_id)

                pad = self.pad_token_id
                attn_ids = (init_ids != pad).long()
                init_emb = self.embedder(torch.cat([init_ids, attn_ids], dim=1)) #This is different from Vec2Text_decoder1.py on line 913 with ...

            corrector_data.append(
                (
                    emb_batch.cpu(),  # true embedding e
                    init_emb.cpu(),  # hypothesis embedding φ(x^(0))
                    init_ids.cpu(),  # initial IDs x^(0)
                    true_labels,  # ground-truth token IDs x
                )
            )

        tr_embs, tr_hyps, tr_ids, tr_labels = zip(*corrector_data)
        tr_embs = torch.cat(tr_embs, dim=0)
        tr_hyps = torch.cat(tr_hyps, dim=0)
        tr_ids = torch.cat(tr_ids, dim=0)
        tr_labels = torch.cat(tr_labels, dim=0)

        corr_train_dataset = TensorDataset(tr_embs, tr_hyps, tr_ids, tr_labels)
        corr_train_loader = DataLoader(
            corr_train_dataset,
            batch_size=corr_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=min(4, cpu_count()),
        )

        # ---- Optional: build Corrector validation dataset ----
        corr_val_loader: Optional[DataLoader] = None
        if val_loader is not None:
            val_corrector_data = []
            for emb_batch, true_labels in val_loader:
                emb_batch = emb_batch.to(self.device, dtype=self.data_type)
                with torch.no_grad():
                    enc_tokens = self.initial_hypothesis.emb_to_seq(emb_batch)
                    init_ids = self.initial_hypothesis.generate(
                        emb_batch, max_length=self.max_length
                    )
                    init_ids_val = self._pad_to_len(
                        init_ids, self.max_length, self.pad_token_id
                    )

                    pad = self.pad_token_id
                    attn_ids_val = (init_ids_val != pad).long()
                    init_emb_val = self.embedder(
                        torch.cat([init_ids_val, attn_ids_val], dim=1)
                    )

                val_corrector_data.append(
                    (emb_batch.cpu(), init_emb_val.cpu(), init_ids_val.cpu(), true_labels)
                )

            if len(val_corrector_data) > 0:
                v_embs, v_hyps, v_ids, v_labels = zip(*val_corrector_data)
                v_embs = torch.cat(v_embs, dim=0)
                v_hyps = torch.cat(v_hyps, dim=0)
                v_ids = torch.cat(v_ids, dim=0)
                v_labels = torch.cat(v_labels, dim=0)

                corr_val_ds = TensorDataset(v_embs, v_hyps, v_ids, v_labels)
                corr_val_loader = DataLoader(
                    corr_val_ds,
                    batch_size=corr_batch_size,
                    shuffle=False,
                    pin_memory=True,
                    num_workers=min(4, cpu_count()),
                )

        # ---- Phase 3: Train Corrector ----
        self.train_loss_per_epoch: List[float] = []
        self.val_loss_per_epoch: List[float] = []
        self.train_val_loss_per_epoch: List[float] = []

        best_val_ce: float = float("inf")
        no_improve_corr: int = 0
        for epoch in range(corr_epochs):
            # Train
            self.corrector.train()
            total_loss, total_count = 0.0, 0

            for emb_batch, hyp_emb, init_ids, labels in corr_train_loader:
                emb_batch = emb_batch.to(self.device, dtype=self.data_type)
                hyp_emb = hyp_emb.to(self.device, dtype=self.data_type)
                init_ids = init_ids.to(self.device)
                labels = labels.to(self.device)

                s = self.num_repeat_tokens
                pad_id = self.pad_token_id
                masked_labels = labels.clone()
                masked_labels[(labels == pad_id) | (labels == pad_idx_data)] = -100

                # Build inputs for the corrector
                real_seq = self.real_Emb2Seq(emb_batch)
                diff_seq = self.difference_Emb2Seq(emb_batch - hyp_emb)
                pred_seq = self.predict_Emb2Seq(hyp_emb)
                token_embs = self.corrector.get_input_embeddings()(init_ids).to(
                    self.data_type
                )

                inputs_embeds = torch.cat(
                    [real_seq, diff_seq, pred_seq, token_embs], dim=1
                )
                pseudo_mask = torch.ones(
                    (inputs_embeds.size(0), 3 * s),
                    device=self.device,
                    dtype=torch.long,
                )
                token_mask = (init_ids != pad_id).long()
                attention_mask = torch.cat([pseudo_mask, token_mask], dim=1)

                outputs = self.corrector(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=masked_labels,
                )
                logits = outputs.logits
                preds = logits[:, :-1, :].float()
                targets = masked_labels[:, 1:]
                loss = F.cross_entropy(
                    preds.reshape(-1, self.vocab_size),
                    targets.reshape(-1),
                    ignore_index=-100,
                    reduction="mean",
                )

                corr_optimizer.zero_grad()
                loss.backward()
                trainable_params = chain(
                    self.corrector.parameters(),
                    self.real_Emb2Seq.parameters(),
                    self.difference_Emb2Seq.parameters(),
                    self.predict_Emb2Seq.parameters(),
                )
                torch.nn.utils.clip_grad_norm_(list(trainable_params), 1.0)
                corr_optimizer.step()

                nonpad = (targets != -100).sum().item()
                total_loss += loss.item() * max(nonpad, 1)
                total_count += nonpad

            avg_train_ce = total_loss / max(1, total_count)
            self.train_loss_per_epoch.append(avg_train_ce)

            # Validation
            avg_val_ce: Optional[float] = None
            if corr_val_loader is not None:
                self.corrector.eval()
                val_loss, val_count = 0.0, 0
                train_val_loss, train_val_count = 0.0, 0
                with torch.no_grad():
                    for emb_batch, hyp_emb, init_ids, labels in corr_val_loader:
                        emb_batch = emb_batch.to(self.device, dtype=self.data_type)
                        hyp_emb = hyp_emb.to(self.device, dtype=self.data_type)
                        init_ids = init_ids.to(self.device)
                        labels = labels.to(self.device)

                        s = self.num_repeat_tokens
                        pad_id = self.pad_token_id
                        masked_labels = labels.clone()
                        masked_labels[(labels == pad_id) | (labels == pad_idx_data)] = -100

                        real_seq = self.real_Emb2Seq(emb_batch)
                        diff_seq = self.difference_Emb2Seq(emb_batch - hyp_emb)
                        pred_seq = self.predict_Emb2Seq(hyp_emb)
                        token_embs = self.corrector.get_input_embeddings()(init_ids).to(
                            self.data_type
                        )

                        inputs_embeds = torch.cat(
                            [real_seq, diff_seq, pred_seq, token_embs], dim=1
                        )
                        pseudo_mask = torch.ones(
                            (inputs_embeds.size(0), 3 * s),
                            device=self.device,
                            dtype=torch.long,
                        )
                        token_mask = (init_ids != pad_id).long()
                        attention_mask = torch.cat([pseudo_mask, token_mask], dim=1)

                        outputs = self.corrector(
                            inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                            labels=masked_labels,
                        )
                        logits = outputs.logits
                        preds = logits[:, :-1, :].float()
                        targets = masked_labels[:, 1:]
                        loss = F.cross_entropy(
                            preds.reshape(-1, self.vocab_size),
                            targets.reshape(-1),
                            ignore_index=-100,
                            reduction="mean",
                        )

                        nonpad = (targets != -100).sum().item()
                        val_loss += loss.item() * max(nonpad, 1)
                        val_count += nonpad

                    if train_eval:
                        for emb_batch, hyp_emb, init_ids, labels in corr_train_loader:
                            emb_batch = emb_batch.to(self.device, dtype=self.data_type)
                            hyp_emb = hyp_emb.to(self.device, dtype=self.data_type)
                            init_ids = init_ids.to(self.device)
                            labels = labels.to(self.device)

                            s = self.num_repeat_tokens
                            pad_id = self.pad_token_id
                            masked_labels = labels.clone()
                            masked_labels[(labels == pad_id) | (labels == pad_idx_data)] = -100

                            real_seq = self.real_Emb2Seq(emb_batch)
                            diff_seq = self.difference_Emb2Seq(emb_batch - hyp_emb)
                            pred_seq = self.predict_Emb2Seq(hyp_emb)
                            token_embs = self.corrector.get_input_embeddings()(init_ids).to(
                                self.data_type
                            )

                            inputs_embeds = torch.cat(
                                [real_seq, diff_seq, pred_seq, token_embs], dim=1
                            )
                            pseudo_mask = torch.ones(
                                (inputs_embeds.size(0), 3 * s),
                                device=self.device,
                                dtype=torch.long,
                            )
                            token_mask = (init_ids != pad_id).long()
                            attention_mask = torch.cat([pseudo_mask, token_mask], dim=1)

                            outputs = self.corrector(
                                inputs_embeds=inputs_embeds,
                                attention_mask=attention_mask,
                                labels=masked_labels,
                            )
                            logits = outputs.logits
                            preds = logits[:, :-1, :].float()
                            targets = masked_labels[:, 1:]
                            loss = F.cross_entropy(
                                preds.reshape(-1, self.vocab_size),
                                targets.reshape(-1),
                                ignore_index=-100,
                                reduction="mean",
                            )

                            nonpad = (targets != -100).sum().item()
                            train_val_loss += loss.item() * max(nonpad, 1)
                            train_val_count += nonpad
                        avg_train_val_ce = train_val_loss / max(1, train_val_count)
                        self.train_val_loss_per_epoch.append(avg_train_val_ce)

                avg_val_ce = val_loss / max(1, val_count)
                self.val_loss_per_epoch.append(avg_val_ce)

                # Optuna pruning
                if trial is not None:
                    step = report_offset + base_epochs + epoch
                    trial.report(float(avg_val_ce if avg_val_ce is not None else avg_train_ce), step=step)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                # --- Early stopping (Corrector) ---
                if avg_val_ce is not None and avg_val_ce < best_val_ce - es_min_delta:
                    best_val_ce = avg_val_ce
                    no_improve_corr = 0
                else:
                    no_improve_corr += 1
                    if no_improve_corr >= es_patience_corr:
                        break

            # Logging
            if avg_val_ce is not None:
                print(
                    f"[Corrector] Epoch {epoch + 1}/{corr_epochs} — "
                    f"Train CE: {avg_train_ce:.4f} — Val CE: {avg_val_ce:.4f}"
                )
            else:
                print(
                    f"[Corrector] Epoch {epoch + 1}/{corr_epochs} — "
                    f"Train CE: {avg_train_ce:.4f}"
                )

        if corr_val_loader is not None and len(self.val_loss_per_epoch) > 0:
            return float(best_val_ce)
        return float(self.train_loss_per_epoch[-1])
