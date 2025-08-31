"""
Utility helpers for GOLLuM featurization (PEP 8 styled).

This module includes:
- Prompt construction for LLMs (:func:`make_template`).
- PEFT target-layer selection for LoRA (:func:`get_target_layers`).
- Dynamic model/tokenizer loading and hidden-size discovery
  (:func:`get_model_tokenizer_dim`).
- Common pooling utilities for transformer hidden states
  (:func:`average_pool`, :func:`last_token_pool`, :func:`weighted_average_pool`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from transformers import AutoModel, AutoTokenizer, T5Config, T5EncoderModel


# === Prompt utility ==========================================================
def make_template(smiles: str) -> str:
    """Construct a text prompt for a molecule to feed into the LLM.

    Parameters
    ----------
    smiles : str
        The molecule SMILES string.

    Returns
    -------
    str
        A formatted string prompt.
    """
    return f"Molecule SMILES: {smiles}"


# === PEFT target layer selection ============================================
def get_target_layers(model: torch.nn.Module, proportion: float = 0.25, from_top: bool = True) -> List[str]:
    """Select a proportion of transformer linear layers for LoRA/PEFT.

    This mirrors the original implementation used in the GOLLuM repository
    (``gollum/featurization/utils/layers.py``). It scans module names to
    infer layer indices compatible with both T5-style (``block.<n>``) and
    BERT-style (``layers.<n>``) encoders, then returns the names of the
    selected ``torch.nn.Linear`` modules.

    Parameters
    ----------
    model : torch.nn.Module
        The transformer model to inspect.
    proportion : float, default=0.25
        Fraction of layers to select in ``(0, 1]``.
    from_top : bool, default=True
        If ``True``, select from the top-most layers; otherwise from bottom.

    Returns
    -------
    list[str]
        Names of selected linear layers.
    """
    all_layers: List[Tuple[int, str]] = []
    layer_numbers: set[int] = set()

    def extract_layer_number(name: str) -> Optional[int]:
        if "block." in name:  # T5 style
            parts = name.split("block.")
            if len(parts) > 1:
                num = parts[1].split(".")[0]
                return int(num) if num.isdigit() else None
        elif "layers." in name:  # BERT style
            parts = name.split("layers.")
            if len(parts) > 1:
                num = parts[1].split(".")[0]
                return int(num) if num.isdigit() else None
        return None

    # First pass: collect all layer numbers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            layer_num = extract_layer_number(name)
            if layer_num is not None:
                layer_numbers.add(layer_num)
                all_layers.append((layer_num, name))

    if not layer_numbers:
        return []

    num_layers = len(layer_numbers)
    num_target_layers = max(1, round(num_layers * proportion))

    sorted_layer_nums = sorted(layer_numbers, reverse=from_top)
    target_layer_nums = set(sorted_layer_nums[:num_target_layers])

    target_modules = [name for layer_num, name in all_layers if layer_num in target_layer_nums]

    print(
        f"\nFound {len(target_modules)} linear layers "
        f"({'top' if from_top else 'bottom'} {proportion*100:.1f}% of {num_layers} layers):"
    )
    print(f"Layer numbers selected: {sorted(target_layer_nums)}")

    return target_modules


# === Model Loader Utility ====================================================
@dataclass
class ModelConfig:
    """Configuration wrapper for supported encoder backbones."""

    name: str
    config_class: Optional[type] = None
    model_class: Optional[type] = None
    dropout_field: str = "dropout_rate"


MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "t5-base": ModelConfig("t5-base", T5Config, T5EncoderModel),
}


def get_model_tokenizer_dim(model_name: str, device: str = "cuda") -> Tuple[Any, Any, int]:
    """Load a pretrained model + tokenizer and return their hidden size.

    This function is adapted from the GOLLuM repo
    (``gollum/featurization/text.py``). If the model is listed in
    :data:`MODEL_CONFIGS`, its config is loaded with dropout disabled
    (configured via ``dropout_field``). Otherwise, a generic
    :class:`~transformers.AutoModel` is loaded.

    Parameters
    ----------
    model_name : str
        Hugging Face model identifier. Must exist in :data:`MODEL_CONFIGS`
        to use the specialized loader; otherwise falls back to
        :class:`~transformers.AutoModel`.
    device : str, default="cuda"
        Device string (e.g., ``"cuda"`` or ``"cpu"``).

    Returns
    -------
    tuple[Any, Any, int]
        ``(model, tokenizer, llm_dim)`` where ``llm_dim`` is the encoder
        hidden size.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    config = None
    if (model_config := MODEL_CONFIGS.get(model_name.lower())) is not None:
        config = model_config.config_class.from_pretrained(model_name)  # type: ignore[union-attr]
        setattr(config, model_config.dropout_field, 0)
        model = model_config.model_class.from_pretrained(model_name, config=config).to(device)  # type: ignore[union-attr]
    else:
        model = AutoModel.from_pretrained(model_name, device_map=device, trust_remote_code=True)
        config = model.config
        print()  # preserved behavior

    if hasattr(config, "hidden_size"):  # type: ignore[truthy-function]
        llm_dim = config.hidden_size  # type: ignore[attr-defined]
    elif hasattr(config, "d_model"):  # type: ignore[truthy-function]
        llm_dim = config.d_model  # type: ignore[attr-defined]
    else:
        raise ValueError(f"Cannot determine embedding dimension for model: {model_name}")

    return model, tokenizer, llm_dim


# === Pooling methods =========================================================
def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Attention-masked average pooling over sequence length.

    Parameters
    ----------
    last_hidden_states : Tensor
        Hidden states of shape ``(B, L, H)``.
    attention_mask : Tensor
        Attention mask of shape ``(B, L)`` where non-zero indicates tokens
        to include.

    Returns
    -------
    Tensor
        Pooled embeddings of shape ``(B, H)``.
    """
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor, left_padding: bool = False) -> Tensor:
    """Select the last (or right-most non-pad) token embedding per sequence.

    Parameters
    ----------
    last_hidden_states : Tensor
        Hidden states of shape ``(B, L, H)``.
    attention_mask : Tensor
        Attention mask of shape ``(B, L)``.
    left_padding : bool, default=False
        If ``True``, assume left padding and take the final position ``L-1``;
        otherwise select the last non-padded token using ``attention_mask``.

    Returns
    -------
    Tensor
        Selected embeddings of shape ``(B, H)``.
    """
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    return last_hidden_states[torch.arange(last_hidden_states.size(0)), sequence_lengths]


def weighted_average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Linearly weighted average pooling over sequence length.

    Weights grow from 1 to ``L`` across the sequence dimension. This mirrors
    the original implementation and keeps the explicit dtype for weights.

    Parameters
    ----------
    last_hidden_states : Tensor
        Hidden states of shape ``(B, L, H)``.
    attention_mask : Tensor
        Attention mask of shape ``(B, L)`` where non-zero indicates tokens
        to include.

    Returns
    -------
    Tensor
        Pooled embeddings of shape ``(B, H)``.
    """
    seq_length = last_hidden_states.size(1)
    weights = (
        torch.arange(1, seq_length + 1, dtype=torch.float32)  # This dtype selection can cause problem in future.
        .unsqueeze(0)
        .to(last_hidden_states.device)
    )

    weighted_mask = weights * attention_mask.float()
    weighted_hidden_states = last_hidden_states * weighted_mask.unsqueeze(-1)

    sum_weighted_embeddings = torch.sum(weighted_hidden_states, dim=1)
    sum_weights = torch.sum(weighted_mask, dim=1, keepdim=True).clamp(min=1)

    weighted_average_embeddings = sum_weighted_embeddings / sum_weights
    return weighted_average_embeddings
