"""Multi-Layer Perceptron for multi-label protein function prediction."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class MultiLayerPerceptron(nn.Module):
    """Parameterised MLP with ReLU activations.

    No sigmoid at the output — designed for use with ``BCEWithLogitsLoss``.

    Args:
        input_dim: Dimensionality of the protein embedding.
        hidden_dims: Sizes of hidden layers (e.g. ``[864, 712]``).
        num_classes: Number of GO-term labels.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        num_classes: int,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
