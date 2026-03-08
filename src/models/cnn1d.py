"""1-D CNN for multi-label protein function prediction."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """Parameterised 1-D CNN operating on protein embeddings.

    The input is reshaped to ``(batch, 1, input_dim)`` before the conv layers.
    No sigmoid at the output — designed for use with ``BCEWithLogitsLoss``.

    Args:
        input_dim: Dimensionality of the protein embedding.
        out_channels: Number of output channels for each conv layer (e.g. ``[3, 8]``).
        kernel_size: Kernel size for all conv layers.
        num_classes: Number of GO-term labels.
    """

    def __init__(
        self,
        input_dim: int,
        out_channels: Sequence[int],
        kernel_size: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        if len(out_channels) < 1:
            raise ValueError("out_channels must have at least one entry")

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=out_channels[0],
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(
            in_channels=out_channels[0],
            out_channels=out_channels[1] if len(out_channels) > 1 else out_channels[0],
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        final_out_ch = out_channels[1] if len(out_channels) > 1 else out_channels[0]
        fc_input_dim = int(final_out_ch * input_dim // 4)

        self.fc1 = nn.Linear(fc_input_dim, 864)
        self.fc2 = nn.Linear(864, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], 1, x.shape[1])
        x = self.pool1(torch.tanh(self.conv1(x)))
        x = self.pool2(torch.tanh(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x
