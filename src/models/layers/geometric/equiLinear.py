import torch
from torch import nn

from src.lib.geometricAlgebraOperations import Blade


class EquiLinearLayer(nn.Module):  # type:ignore[misc]
    def __init__(self, inputFeatures: int, outputFeatures: int) -> None:
        super(EquiLinearLayer, self).__init__()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.equiLinBasis = Blade.getEquiLinBasis(self.device)

        self.weights = nn.Parameter(
            torch.rand(
                outputFeatures,
                inputFeatures,
                self.equiLinBasis.shape[0],
                device=self.device,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        o: output features
        i: input features
        b: equinLinBasis number (9)
        g: Geometric Algebra size (16)
        """

        return torch.einsum(
            "oib, bgg, ...ig -> ...og", self.weights, self.equiLinBasis, x
        )
