"""
GRU-based sequence decoder (PEP 8 styled).

A PyTorch reimplementation of the decoder used in the VAE architecture of
Gomez-Bombarelli et al. This module provides two components:

- :class:`TerminalGRU`: an autoregressive GRUCell decoder that supports teacher
  forcing during training and sampling during inference.
- :class:`GRU`: a higher-level decoder that applies dense projections, an
  optional GRU stack, and a final decoding head (either :class:`TerminalGRU`
  or a plain GRU + linear projection).
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TerminalGRU(nn.Module):
    """Autoregressive GRUCell decoder with teacher forcing.

    During training, the module consumes the previous *ground-truth* token
    (teacher forcing). During evaluation/inference, it autoregressively samples
    the next token from its own softmax outputs.

    Parameters
    ----------
    input_size : int
        Dimensionality of the per-timestep input features.
    hidden_size : int
        Size of the GRUCell hidden state.
    num_classes : int
        Vocabulary size for the categorical outputs.
    dropout : float, default=0.0
        Dropout probability applied to the concatenated input.
    temperature : float, default=1.0
        Softmax temperature used during sampling at inference (>= 0).
    seed : int, default=42
        Random seed for reproducibility (sets :func:`torch.manual_seed`).
    bos_idx : int | None, default=None
        Optional index of the BOS token used to initialize decoding.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        dropout: float = 0.0,
        temperature: float = 1.0,
        seed: int = 42,
        bos_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.temperature = temperature
        self.bos_idx = bos_idx
        torch.manual_seed(seed)

        # GRU gates (z, r, h) with concatenated feature + one-hot(previous token)
        self.gru_cell = nn.GRUCell(input_size + num_classes, hidden_size)
        # Linear map for previous one-hot token (behavior preserved)
        self.recurrent_kernel_y = nn.Linear(num_classes, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, gru_inputs: Tensor, true_seq: Optional[Tensor] = None) -> Tensor:
        """Decode a sequence of logits over the vocabulary.

        Parameters
        ----------
        gru_inputs : Tensor
            Tensor of shape ``(B, T, input_size)`` containing per-step features.
        true_seq : Tensor | None, default=None
            If provided during training, a one-hot tensor of shape
            ``(B, T, num_classes)`` used for teacher forcing, where the
            decoder consumes ``true_seq[:, t-1]`` at step ``t``.

        Returns
        -------
        Tensor
            Logits of shape ``(B, T, num_classes)`` (apply softmax externally).
        """
        batch_size, timesteps, _ = gru_inputs.size()
        h = torch.zeros(batch_size, self.hidden_size, device=gru_inputs.device)

        outputs = []
        for t in range(timesteps):
            x_t = gru_inputs[:, t]  # (B, input_size)

            if self.training:
                if t == 0:
                    # Start from BOS if available; otherwise a zero one-hot
                    if self.bos_idx is not None:
                        prev_ids = torch.full(
                            (batch_size,), self.bos_idx, device=gru_inputs.device
                        )
                        prev_y = F.one_hot(prev_ids, num_classes=self.num_classes).float()
                    else:
                        prev_y = torch.zeros(
                            batch_size, self.num_classes, device=gru_inputs.device
                        )
                else:
                    prev_y = true_seq[:, t - 1]
            else:
                if t == 0 and self.bos_idx is not None:
                    prev_ids = torch.full(
                        (batch_size,), self.bos_idx, device=gru_inputs.device
                    )
                else:
                    logits = self.output_layer(h) / self.temperature
                    probs = F.softmax(logits, dim=-1)
                    prev_ids = probs.multinomial(1).squeeze(-1)
                prev_y = F.one_hot(prev_ids, num_classes=self.num_classes).float()

            # Concatenate per-step features with previous one-hot token
            gru_input = torch.cat([x_t, prev_y], dim=-1)
            h = self.gru_cell(self.dropout(gru_input), h)

            logits = self.output_layer(h)
            outputs.append(logits)

        return torch.stack(outputs, dim=1)  # (B, T, num_classes)


class GRU(nn.Module):
    """Stacked dense + GRU decoder with configurable final head.

    The module first applies a stack of fully-connected layers to expand the
    input embedding, optionally applies a GRU stack for temporal refinement,
    and finally decodes with either a :class:`TerminalGRU` head (teacher
    forcing + sampling) or a plain GRU followed by a linear projection.

    Parameters
    ----------
    params : dict
        Hyperparameter dictionary with the following keys:

        - ``hidden_dim`` (int): input embedding dimension.
        - ``middle_layer`` (int): number of dense layers in the middle block.
        - ``hg_growth_factor`` (float): geometric growth factor for dense widths.
        - ``activation`` (str | callable): activation in the middle block
          (if str, a function is fetched from :mod:`torch.nn.functional`).
        - ``dropout_rate_mid`` (float): dropout probability in the middle block.
        - ``batchnorm_mid`` (bool): whether to apply BatchNorm1d after each dense.
        - ``MAX_LEN`` (int): sequence length to decode.
        - ``NCHARS`` (int): vocabulary size.
        - ``gru_depth`` (int): total GRU depth including final head.
        - ``recurrent_dim`` (int): hidden size of the GRU layers.
        - ``do_tgru`` (bool): if ``True``, use :class:`TerminalGRU` as head;
          otherwise use a plain GRU + Linear projection.
        - ``tgru_dropout`` (float, optional): dropout for :class:`TerminalGRU`.
        - ``temperature`` (float, optional): sampling temperature for head.
        - ``RAND_SEED`` (int, optional): random seed passed to head.
    bos_idx : int | None, default=None
        Optional BOS token index forwarded to :class:`TerminalGRU`.
    """

    def __init__(self, params: Dict[str, Any], bos_idx: Optional[int] = None) -> None:
        super().__init__()
        self.hidden_dim = params["hidden_dim"]
        self.middle_layer = params["middle_layer"]
        self.hg_growth = params["hg_growth_factor"]
        self.activation = (
            getattr(F, params["activation"])
            if isinstance(params["activation"], str)
            else params["activation"]
        )
        self.dropout_rate = params["dropout_rate_mid"]
        self.batchnorm_mid = params["batchnorm_mid"]
        self.MAX_LEN = params["MAX_LEN"]
        self.NCHARS = params["NCHARS"]
        self.gru_depth = params["gru_depth"]
        self.recurrent_dim = params["recurrent_dim"]
        self.do_tgru = params["do_tgru"]
        self.bos_idx = bos_idx

        # --- Dense middle layers ---
        layers = []
        # First up-projection
        layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        for i in range(1, self.middle_layer):
            in_dim = int(self.hidden_dim * (self.hg_growth ** (i - 1)))
            out_dim = int(self.hidden_dim * (self.hg_growth ** i))
            layers.append(nn.Linear(in_dim, out_dim))
        self.denses = nn.ModuleList(layers)

        if self.batchnorm_mid:
            self.bns = nn.ModuleList(
                [
                    nn.BatchNorm1d(int(self.hidden_dim * (self.hg_growth ** i)))
                    for i in range(self.middle_layer)
                ]
            )
        else:
            self.bns = [None] * self.middle_layer  # type: ignore[assignment]

        self.dropout = (
            nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()
        )

        # --- Recurrent stack (before final head) ---
        if self.gru_depth > 1:
            self.gru_stack = nn.GRU(
                input_size=int(self.hidden_dim * (self.hg_growth ** (self.middle_layer - 1))),
                hidden_size=self.recurrent_dim,
                num_layers=self.gru_depth - 1,
                batch_first=True,
            )
        else:
            self.gru_stack = None

        # --- Final layer: either TerminalGRU or plain GRU head ---
        self.last_dense_dim = int(self.hidden_dim * (self.hg_growth ** (self.middle_layer - 1)))
        final_in_dim = self.recurrent_dim if self.gru_depth > 1 else self.last_dense_dim

        if self.do_tgru:
            # Note: TerminalGRU expects feature-dim == recurrent_dim
            self.final = TerminalGRU(
                input_size=final_in_dim,
                hidden_size=self.recurrent_dim,
                num_classes=self.NCHARS,
                dropout=params.get("tgru_dropout", 0.0),
                temperature=params.get("temperature", 1.0),
                seed=params.get("RAND_SEED", 42),
                bos_idx=self.bos_idx,
            )
        else:
            self.final_gru = nn.GRU(
                input_size=final_in_dim,
                hidden_size=self.recurrent_dim,
                num_layers=1,
                batch_first=True,
            )
            self.final_proj = nn.Linear(self.recurrent_dim, self.NCHARS)

    def forward(self, x: Tensor, true_seq: Optional[Tensor] = None) -> Tensor:
        """Decode a sequence of logits from a latent embedding.

        Parameters
        ----------
        x : Tensor
            Input embedding of shape ``(B, hidden_dim)``.
        true_seq : Tensor | None, default=None
            Ground-truth token indices ``(B, MAX_LEN)`` or one-hot
            encodings ``(B, MAX_LEN, NCHARS)`` used by
            :class:`TerminalGRU` during training.

        Returns
        -------
        Tensor
            Logits of shape ``(B, MAX_LEN, NCHARS)``.
        """
        # Dense middle projection with activation, dropout, optional batchnorm
        for idx, dense in enumerate(self.denses):
            x = dense(x)
            x = self.activation(x)
            x = self.dropout(x)
            if self.bns[idx] is not None:
                x = self.bns[idx](x)  # (B, C) -> (B, C)

        # Tile to sequence: (B, hidden) -> (B, MAX_LEN, hidden)
        x = x.unsqueeze(1).repeat(1, self.MAX_LEN, 1)

        # Recurrent refinement
        if self.gru_stack is not None:
            x, _ = self.gru_stack(x)  # (B, MAX_LEN, recurrent_dim)

        # Final projection
        if self.do_tgru:
            # Ensure true_seq is one-hot during training if loader gives indices
            if self.training and (true_seq is not None) and (true_seq.dim() == 2):
                true_seq = F.one_hot(true_seq, num_classes=self.NCHARS).float()
            logits = self.final(x, true_seq)  # (B, T, NCHARS)
        else:
            out, _ = self.final_gru(x)  # (B, T, recurrent_dim)
            logits = self.final_proj(out)  # (B, T, NCHARS)

        return logits
