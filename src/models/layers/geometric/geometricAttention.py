import torch
from torch import nn


class GeometricAttentionLayer(nn.Module):  # type:ignore[misc]
    def __init__(self) -> None:
        super(GeometricAttentionLayer, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
