"""BHI-style parallel-CNN plus bidirectional-LSTM regressor.

Purpose:
    Reproduce the architecture used throughout the DeBour experiments: parallel
    9/15 bp convolutions, a BiLSTM, parallel 9/15 bp core convolutions, a 1x1
    mapper, global average pooling, and one unconstrained activity output.

How to use:
    ``model = BhiCnnBiLstm(dropout=0.1)`` and pass tensors shaped
    ``(batch, 4, sequence_length)``. Run this module to print parameter count.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding="same")
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.dropout(F.relu(self.conv(inputs)))


class BhiCnnBiLstm(nn.Module):
    """Regression model matching the historical 6.8M-parameter configuration."""

    def __init__(
        self,
        dropout: float = 0.1,
        stem_channels: int = 512,
        lstm_hidden_channels: int = 438,
        core_channels: int = 320,
    ):
        super().__init__()
        self.first_conv_9 = ConvBlock(4, stem_channels // 2, 9, dropout)
        self.first_conv_15 = ConvBlock(4, stem_channels // 2, 15, dropout)
        self.lstm = nn.LSTM(stem_channels, lstm_hidden_channels, batch_first=True, bidirectional=True)
        self.core_conv_9 = ConvBlock(2 * lstm_hidden_channels, core_channels // 2, 9, dropout)
        self.core_conv_15 = ConvBlock(2 * lstm_hidden_channels, core_channels // 2, 15, dropout)
        self.core_dropout = nn.Dropout(dropout)
        self.mapper = nn.Conv1d(core_channels, 256, kernel_size=1, padding="same")
        self.output = nn.Linear(256, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                std = math.sqrt(2.0 / (module.kernel_size[0] * module.out_channels))
                module.weight.data.normal_(0.0, std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                module.weight.data.normal_(0.0, 0.001)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = torch.cat([self.first_conv_9(inputs), self.first_conv_15(inputs)], dim=1)
        hidden, _ = self.lstm(hidden.permute(0, 2, 1))
        hidden = hidden.permute(0, 2, 1)
        hidden = torch.cat([self.core_conv_9(hidden), self.core_conv_15(hidden)], dim=1)
        hidden = self.mapper(self.core_dropout(hidden))
        hidden = F.adaptive_avg_pool1d(hidden, 1).squeeze(-1)
        return self.output(hidden).squeeze(-1)


if __name__ == "__main__":
    model = BhiCnnBiLstm()
    print(f"Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    print("Output shape:", tuple(model(torch.zeros(2, 4, 200)).shape))

