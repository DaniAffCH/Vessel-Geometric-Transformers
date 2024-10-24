from torch import Tensor, nn, optim

from config.dataclasses import BaselineConfig
from src.data.dataset import NUM_FEATURES
from src.lib.geometricAlgebraElements import GeometricAlgebraBase
from src.models.base_model import VesselClassificationModel
from src.models.layers.default.transformer import TransformerEncoder


class BaselineTransformer(VesselClassificationModel):
    """
    Baseline Transformer model for classification.

    Args:
        config (BaselineConfig): Configuration object
                                 for setting model parameters.
    """

    def __init__(self, config: BaselineConfig) -> None:
        super().__init__()

        self.encoder = TransformerEncoder(
            embed_dim=config.transformer_embedding_dim,
            num_heads=config.transformer_num_heads,
            feedforward_dim=config.transformer_feedforward_dim,
            num_layers=config.transformer_num_layers,
        )

        self.embedder = nn.Linear(
            GeometricAlgebraBase.GA_size, config.transformer_embedding_dim
        )

        self.projection = nn.Linear(
            config.transformer_embedding_dim
            * config.features_size_limit
            * NUM_FEATURES,
            1,
        )
        self.config = config

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        # (batch_size, num_elements*seq_length, ga_size)
        x = x.reshape(x.size(0), -1, x.size(-1))

        # (batch_size, num_elements*seq_length)
        mask = mask.reshape(x.size(0), -1)

        x = self.embedder(x)
        x = self.encoder(x, mask)

        x = x.reshape(x.size(0), -1)

        x = self.projection(x)

        return x  # Logits are used directly for BCEWithLogitsLoss

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters(), lr=self.config.learning_rate)
