import torch
import torch.nn.functional as F
from torch import nn


class GatedGELU(nn.Module):  # type:ignore[misc]
    """
    Applies the GeLU only to the scalar part of the multivector
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x[..., [0]]) * x
