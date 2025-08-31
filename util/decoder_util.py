"""
Utility functions for decoding model outputs (PEP 8 styled).

This module provides helpers to:
- Convert batched model outputs (one-hot logits or label IDs) into SELFIES or
  SMILES strings: :func:`convert_to_string`.
- Compute class weights for cross-entropy using the Effective Number of Samples
  heuristic (Cui et al., 2019): :func:`compute_effective_num_weights`.
"""

from typing import Any, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn
from selfies.decoder import decoder
from selfies.utils.encoding_utils import encoding_to_selfies
from torch import Tensor


def convert_to_string(
    data: Union[Tensor, np.ndarray, Sequence[Sequence[int]]],
    id_to_token: Mapping[int, str],
    input_data_type: Literal["label", "one-hot"],
    to_smiles: bool = True,
    eos_token: Optional[str] = "[eos]",
    pad_token: Optional[str] = "[nop]",
    bos_token: Optional[str] = "[bos]",
) -> List[Optional[str]]:
    """Convert batched outputs to SELFIES or SMILES strings.

    Parameters
    ----------
    data
        If ``input_data_type == 'one-hot'``: shape ``(B, L, V)`` (logits/probs/one-hot).
        If ``input_data_type == 'label'``:   shape ``(B, L)`` integer token IDs.
    id_to_token
        Mapping from int id → SELFIES token (0..V-1 contiguous indices).
    input_data_type
        Either ``'label'`` or ``'one-hot'``.
    to_smiles
        If ``True``, convert SELFIES → SMILES using :mod:`selfies.decoder`.
        Otherwise, return SELFIES strings.
    eos_token
        Optional SELFIES token string at which to truncate (excluded).
    pad_token
        Optional SELFIES token string to remove anywhere.
    bos_token
        Optional SELFIES BOS token string to remove anywhere.

    Returns
    -------
    list[Optional[str]]
        Per-example decoded strings (SMILES if ``to_smiles=True``, else SELFIES).
        ``None`` for sequences that fail to decode.
    """
    # ---- Normalize input to a torch tensor on CPU ----
    if isinstance(data, torch.Tensor):
        tensor = data.detach().to("cpu")
    elif isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    else:
        tensor = torch.as_tensor(data)

    # ---- Convert to label IDs (B, L) regardless of input type ----
    if tensor.ndim == 3:
        # (B, L, V) -> argmax over vocab
        labels = tensor.argmax(dim=-1)
    elif tensor.ndim == 2:
        if input_data_type == "one-hot":
            raise ValueError(
                "One-hot input must be 3D (Batch size, Sequence length, Vocab size). Got 2D."
            )
        labels = tensor.long() if tensor.dtype not in (torch.long, torch.int64) else tensor
    else:
        raise ValueError(
            f"Unsupported input shape {tuple(tensor.shape)}; expected 2D (B, L) or 3D (B, L, V)."
        )

    # ---- Prepare PAD/EOS/BOS handling ----
    token_to_id = {tok: idx for idx, tok in id_to_token.items()}
    eos_id = token_to_id.get(eos_token) if eos_token is not None else None
    pad_id = token_to_id.get(pad_token) if pad_token is not None else None
    bos_id = token_to_id.get(bos_token) if bos_token is not None else None

    results: List[Optional[str]] = []
    for seq in labels:  # (L,)
        seq_ids = seq.tolist()

        # Remove BOS/PAD and truncate at EOS if provided
        if bos_id is not None:
            seq_ids = [i for i in seq_ids if i != bos_id]
        if pad_id is not None:
            seq_ids = [i for i in seq_ids if i != pad_id]
        if eos_id is not None and eos_id in seq_ids:
            seq_ids = seq_ids[: seq_ids.index(eos_id)]

        # IDs -> SELFIES (per sequence)
        try:
            selfies_str = encoding_to_selfies(
                encoding=seq_ids,
                vocab_itos=id_to_token,
                enc_type="label",
            )
        except Exception:
            results.append(None)
            continue

        # Optionally SELFIES -> SMILES
        if to_smiles:
            try:
                results.append(decoder(selfies_str))
            except Exception:
                results.append(None)
        else:
            results.append(selfies_str)

    return results


@torch.no_grad()
def compute_effective_num_weights(
    train_loader: Iterable[Tuple[Any, Tensor]],
    vocab_size: int,
    pad_idx: int,
    eos_idx: Optional[int] = None,
    bos_idx: Optional[int] = None,
    beta: float = 0.999,
    clip_range: Tuple[float, float] = (0.2, 5.0),
    device: torch.device = torch.device("cpu"),
) -> torch.FloatTensor:
    """Compute class weights using Effective Number of Samples (Cui et al.).

    The weights can be passed to :class:`torch.nn.CrossEntropyLoss` via the
    ``weight=...`` argument. PAD is ignored by CE via ``ignore_index=pad_idx``;
    we set its weight to 0 for clarity. EOS/BOS are kept neutral (1.0) when
    indices are provided.

    Parameters
    ----------
    train_loader
        Iterable yielding ``(inputs, labels)`` where labels are IDs ``(B, T)``
        or one-hot ``(B, T, V)``; only labels are used.
    vocab_size
        Vocabulary size (``V``).
    pad_idx
        Index of PAD token (ignored in loss).
    eos_idx
        Optional EOS token index to keep neutral (weight = 1.0).
    bos_idx
        Optional BOS token index to keep neutral (weight = 1.0).
    beta
        Decay factor in ``(1 - beta) / (1 - beta**f_c)``; closer to 1.0 for
        larger corpora (typical range: 0.99–0.9999).
    clip_range
        Min/max clip applied after normalizing weights to mean 1.0.
    device
        Device for internal counting tensors.

    Returns
    -------
    torch.FloatTensor
        Weight vector ``w`` of shape ``(vocab_size,)`` on CPU.
    """
    counts = torch.zeros(vocab_size, dtype=torch.long, device=device)

    for _, y in train_loader:
        # y: (B, T) int labels OR (B, T, V) one-hot → convert to IDs
        if isinstance(y, torch.Tensor) and y.dim() == 3:
            y = y.argmax(dim=-1)
        y = y.to(device)
        # exclude PAD from counting
        mask = y != pad_idx
        ids = y[mask].reshape(-1)
        if ids.numel():
            counts += torch.bincount(ids, minlength=vocab_size)

    freqs = counts.to(torch.float32)  # f_c
    # Effective number (Cui et al. 2019): w_c ∝ (1 - beta) / (1 - beta^{f_c})
    # Use f_c>=1 to avoid division by zero; set w_c=0 for unseen classes
    denom = 1.0 - torch.pow(beta, torch.clamp(freqs, min=1.0))
    weights = (1.0 - beta) / denom
    weights[freqs == 0] = 0.0

    # Normalize weights to mean 1.0 for stable LR, then clip
    nz = weights > 0
    mean_w = weights[nz].mean().clamp(min=1e-8)
    weights = weights / mean_w
    wmin, wmax = clip_range
    weights = weights.clamp_(min=wmin, max=wmax)

    # Housekeeping: PAD is ignored by CE anyway; set weight=0. EOS/BOS neutral
    weights[pad_idx] = 0.0
    if eos_idx is not None and 0 <= eos_idx < vocab_size:
        weights[eos_idx] = 1.0
    if bos_idx is not None and 0 <= bos_idx < vocab_size:
        weights[bos_idx] = 1.0

    return weights.cpu()

def build_adamW(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    """
    Create an AdamW optimizer with proper parameter grouping:
    - Apply weight decay to 'true' weights (Linear, Embedding, GRU matrices).
    - Do NOT decay biases or normalization parameters (BatchNorm/LayerNorm).

    Args:
        model: The model to optimize.
        lr: Learning rate.
        weight_decay: Weight decay coefficient (L2 strength).
        betas: AdamW betas.
        eps: AdamW epsilon.

    Returns:
        Configured AdamW optimizer.
    """
    decay: List[torch.nn.Parameter] = []
    no_decay: List[torch.nn.Parameter] = []

    whitelist_weight_modules = (nn.Linear, nn.Embedding)
    norm_modules = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue

            full_name = f"{module_name}.{param_name}" if module_name else param_name

            # Biases: never decay
            if param_name.endswith("bias"):
                no_decay.append(param)
                continue

            # Normalization layers: never decay
            if isinstance(module, norm_modules):
                no_decay.append(param)
                continue

            # Recurrent layer weights (GRU/LSTM) are named weight_ih_l*, weight_hh_l*
            if "weight_ih" in param_name or "weight_hh" in param_name:
                decay.append(param)
                continue

            # Linear/Embedding weights: decay
            if isinstance(module, whitelist_weight_modules) and param_name == "weight":
                decay.append(param)
                continue

            # Fallback: if 1D (e.g., biases, norm scales) do not decay; else decay
            (no_decay if param.ndim == 1 else decay).append(param)

    param_groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps)
    return optimizer