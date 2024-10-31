import torch
from torch import nn

from src.lib.geometricAlgebraOperations import Blade


class EquiLinearLayer(nn.Module):  # type:ignore[misc]
    """
    Computes a linear transformation in the Geometric Algebra space.
    """

    def __init__(self, inputFeatures: int, outputFeatures: int) -> None:
        super(EquiLinearLayer, self).__init__()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.equiLinBasis = Blade.getEquiLinBasis()

        self.weights = nn.Parameter(
            torch.empty(
                (outputFeatures, inputFeatures, self.equiLinBasis.shape[0]),
                device=self.device,
            )
        )
        nn.init.xavier_uniform_(self.weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (..., input features) in GA space

        Returns:
            output tensor of shape (..., output features) in GA space

        o: output features i: input features b: equinLinBasis number (9) g:
        Geometric Algebra size (16)

        The product is performed by multiplying the input tensor with the
        precomputed equiLinBasis and the weights tensor.
        """

        return torch.einsum(
            "oib, bgg, ...ig -> ...og", self.weights, self.equiLinBasis, x
        )
