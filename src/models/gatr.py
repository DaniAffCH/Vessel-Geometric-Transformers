import torch
from torch import Tensor, nn, optim

from config.dataclasses import GatrConfig
from src.data.dataset import NUM_FEATURES
from src.lib.geometricAlgebraElements import GeometricAlgebraBase
from src.models.base_model import VesselClassificationModel
from src.models.layers.geometric import EquiLinearLayer, GATrBlock


class Gatr(VesselClassificationModel):
    """
    Geometric Algebra Transformer model for classification.

    Args:
        config (GatrConfig): Configuration object
                                 for setting model parameters.
    """

    def __init__(self, config: GatrConfig) -> None:
        super(Gatr, self).__init__()

        self.hsProjection = EquiLinearLayer(NUM_FEATURES, config.hidden_size)

        self.backbone = nn.ModuleList(
            [
                GATrBlock(config.hidden_size, config.num_attention_heads)
                for _ in range(config.num_backbone_layers)
            ]
        )

        self.finalProjection = nn.Linear(
            config.hidden_size
            * config.features_size_limit
            * GeometricAlgebraBase.GA_size,
            1,
        )

        self.config = config

    def getReference(self, x: Tensor) -> torch.Tensor:
        # shape (batch, 1, ..., 1, 16)
        dim = tuple(range(1, len(x.shape) - 1))
        return torch.mean(x, dim=dim, keepdim=True)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = x.transpose(-2, -3)

        reference = self.getReference(x)

        x = self.hsProjection(x)

        for layer in self.backbone:
            x = layer(x, reference)

        x = x.reshape(x.shape[0], -1)

        x = self.finalProjection(x)

        return x  # Logits are used directly for BCEWithLogitsLoss

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters(), lr=self.config.learning_rate)
