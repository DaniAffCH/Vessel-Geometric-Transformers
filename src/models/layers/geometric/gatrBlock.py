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
    def __init__(self, hidden_size: int, num_attention_heads: int) -> None:
        super(GATrBlock, self).__init__()

        self.branch1 = nn.ModuleList(
            [
                EquiNormLayer(),
                EquiLinearLayer(hidden_size, hidden_size),
                GeometricAttentionLayer(hidden_size, num_attention_heads),
                EquiLinearLayer(hidden_size, hidden_size),
            ]
        )

        self.branch2 = nn.ModuleList(
            [
                EquiNormLayer(),
                EquiLinearLayer(hidden_size, hidden_size),
                GeometricBilinearLayer(hidden_size, hidden_size),
                GatedGELU(),
                EquiLinearLayer(hidden_size, hidden_size),
            ]
        )

    def forward(
        self, x: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:

        xb = x
        for lf in self.branch1:
            if isinstance(lf, GeometricAttentionLayer):
                xb = lf(xb, xb, xb)
            else:
                xb = lf(xb)

        x = xb + x
        xb = x

        for ls in self.branch2:
            if isinstance(ls, GeometricAttentionLayer):
                xb = ls(xb, xb, xb)
            elif isinstance(ls, GeometricBilinearLayer):
                xb = ls(xb, reference)
            else:
                xb = ls(xb)

        x = xb + x

        return x
