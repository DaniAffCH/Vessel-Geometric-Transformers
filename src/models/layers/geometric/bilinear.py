import torch
from torch import nn

from src.lib.geometricAlgebraOperations import (
    EquivariantJoin,
    GeometricProduct,
)
from src.models.layers.geometric.equiLinear import EquiLinearLayer


class GeometricBilinearLayer(nn.Module):  # type:ignore[misc]
    """
    Computes a bilinear transformation in the Geometric Algebra space. This
    layer concatenates the geometric product and the join of the input
    tensors, after applying an equilinear transformation to the input
    multivectors. The final output is then projected again through an
    equilinear transformation.
    """

    def __init__(self, inputFeatures: int, outputFeatures: int) -> None:
        super(GeometricBilinearLayer, self).__init__()

        # Ensure the number of output features is even for correct splitting
        assert outputFeatures % 2 == 0, "Output features number must be even"

        hiddenFeatures = outputFeatures // 2

        # Separate projections for product and join components
        self.prodX_proj = EquiLinearLayer(inputFeatures, hiddenFeatures)
        self.prodY_proj = EquiLinearLayer(inputFeatures, hiddenFeatures)
        self.joinX_proj = EquiLinearLayer(inputFeatures, hiddenFeatures)
        self.joinY_proj = EquiLinearLayer(inputFeatures, hiddenFeatures)

        self.final_proj = EquiLinearLayer(inputFeatures, outputFeatures)

    def forward(
        self, x: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        # Compute product features
        prodX = self.prodX_proj(x)
        prodY = self.prodY_proj(x)
        prodFeatures = GeometricProduct.apply(prodX, prodY)

        # Compute join features, using reference for equivariance
        joinX = self.joinX_proj(x)
        joinY = self.joinY_proj(x)
        joinFeatures = EquivariantJoin.apply(joinX, joinY, reference)

        # Concatenate product and join features along the last dimension
        features = torch.cat((prodFeatures, joinFeatures), dim=-2)

        # Final projection to output feature space
        features = self.final_proj(features)

        return features
