import torch
from torch import nn

from src.lib.geometricAlgebraOperations import InnerProduct


class EquiNormLayer(nn.Module):  # type:ignore[misc]
    """
    Normalizes the input tensor in the Geometric Algebra space by dividing
    the input tensor by the square root of the average of the inner product
    over the batch.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # needed to avoid division by 0
        epsilon = 1e-8
        ev = torch.mean(InnerProduct.apply(x, x), dim=-2, keepdim=True)
        ev = ev.unsqueeze(-1)

        return x / torch.sqrt(ev + epsilon)
