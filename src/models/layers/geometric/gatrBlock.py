import torch
from torch import nn

from src.models.layers.geometric import (
    EquiLinearLayer,
    EquiNormLayer,
    GatedGELU,
    GeometricBilinearLayer,
)

from .geometricAttention import GeometricAttentionLayer


class GATrBlock(nn.Module):  # type:ignore[misc]
    def __init__(self, hidden_size: int) -> None:
        super(GATrBlock, self).__init__()

        self.branch1 = nn.Sequential(
            EquiNormLayer(),
            EquiLinearLayer(hidden_size, hidden_size),
            GeometricAttentionLayer(),
            EquiLinearLayer(hidden_size, hidden_size),
        )

        self.branch2 = nn.Sequential(
            EquiNormLayer(),
            EquiLinearLayer(hidden_size, hidden_size),
            GeometricBilinearLayer(hidden_size, hidden_size),
            GatedGELU(),
            EquiLinearLayer(hidden_size, hidden_size),
        )

    def forward(
        self, x: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        xb1 = self.branch1(x) + x

        x = self.branch2[0:2](xb1)
        x = self.branch2[2](x, reference)
        x = self.branch2[3:](x)

        x = xb1 + x

        return x
