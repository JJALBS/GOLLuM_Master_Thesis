"""
Minimal MLP decoder (PEP 8 styled).

This module defines a very simple multilayer perceptron (MLP) decoder for
mapping a fixed-size input embedding to a sequence of categorical logits.
"""

from typing import Tuple

import torch.nn as nn
from torch import Tensor


class SimpleMLP(nn.Module):
    """A compact MLP decoder with configurable depth and dropout.

    The network applies ``depth`` repeated blocks of ``Linear -> ReLU -> (Dropout)``
    followed by a final linear layer that maps to the flattened output size.
    The output is reshaped to ``(B, output_dim[0], output_dim[1])``.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input embedding.
    hidden_dim : int
        Width of the hidden layers.
    output_dim : tuple[int, int]
        Target output shape (``rows``, ``cols``); the product determines the
        output layer size.
    depth : int
        Number of ``Linear -> ReLU -> (Dropout)`` blocks.
    dropout : float
        Dropout probability applied after each ReLU if ``dropout > 0``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: Tuple[int, int],
        depth: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.depth = depth
        self.dropout = dropout

        layers = []
        output_dim_flat = output_dim[0] * output_dim[1]

        for _ in range(depth):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, output_dim_flat))
        self.structure = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Compute logits and reshape to the requested output dimensions.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape ``(B, input_dim)``.

        Returns
        -------
        Tensor
            Output tensor of shape ``(B, output_dim[0], output_dim[1])``.
        """
        flat = self.structure(x)
        batch_size = x.shape[0]
        result = flat.view(batch_size, *self.output_dim)
        return result
