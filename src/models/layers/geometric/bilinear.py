import torch
from torch import nn

from src.lib.geometricAlgebraOperations import (
    EquivariantJoin,
    GeometricProduct,
)
from src.models.layers.geometric.equiLinear import EquiLinearLayer


class GeometricBilinearLayer(nn.Module):  # type:ignore[misc]
    def __init__(self, inputFeatures: int, outputFeatures: int) -> None:
        super(GeometricBilinearLayer, self).__init__()

        assert outputFeatures % 2 == 0, "Output features number must be even"

        hiddenFeatures = outputFeatures // 2

        self.prodX_proj = EquiLinearLayer(inputFeatures, hiddenFeatures)
        self.prodY_proj = EquiLinearLayer(inputFeatures, hiddenFeatures)
        self.joinX_proj = EquiLinearLayer(inputFeatures, hiddenFeatures)
        self.joinY_proj = EquiLinearLayer(inputFeatures, hiddenFeatures)

        self.final_proj = EquiLinearLayer(inputFeatures, outputFeatures)

    def forward(
        self, x: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:

        prodX = self.prodX_proj(x)
        prodY = self.prodY_proj(x)

        prodFeatures = GeometricProduct.apply(prodX, prodY)

        joinX = self.joinX_proj(x)
        joinY = self.joinY_proj(x)

        joinFeatures = EquivariantJoin.apply(joinX, joinY, reference)

        features = torch.cat((prodFeatures, joinFeatures), dim=-2)

        features = self.final_proj(features)

        return features
