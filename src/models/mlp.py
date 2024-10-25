import einops
from torch import Tensor, nn, optim

from config.dataclasses import MLPConfig
from src.data.dataset import NUM_FEATURES
from src.models.base_model import VesselClassificationModel


class BaselineMLP(VesselClassificationModel):
    """
    Baseline Transformer model for classification.

    Args:
        config (BaselineConfig): Configuration object
                                 for setting model parameters.
    """

    def __init__(self, config: MLPConfig) -> None:
        super().__init__()
        self.config = config

        self.fc = nn.Sequential(
            nn.Linear(
                NUM_FEATURES * self.config.features_size_limit * 16,
                config.hidden_size,
            ),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 1),
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        # (batch_size, num_elements*seq_length*ga_size)
        x = einops.rearrange(x, "b f d g -> b (f d g)")
        logits = self.fc(x)
        return logits  # Logits are used directly for BCEWithLogitsLoss

    def configure_optimizers(self) -> optim.Optimizer:
        """
        Configure the optimizer.

        Returns:
            optim.Optimizer: The Adam optimizer.
        """
        return optim.Adam(self.parameters(), lr=self.config.learning_rate)
