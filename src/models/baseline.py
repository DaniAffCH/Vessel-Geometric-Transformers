import lightning as L
from torch import Tensor, nn, optim

from config.dataclasses import BaselineConfig
from src.data.datamodule import VesselBatch
from src.models.layers.default.transformer import TransformerEncoder


class BaselineTransformer(L.LightningModule):  # type: ignore[misc]
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
        self.projection = nn.Linear(config.transformer_embedding_dim, 1)
        self.config = config
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Forward pass through the model.

        Args:
            x (Tensor): Input tensor of shape
                        (batch_size, seq_length, embed_dim).
            mask (Tensor): Mask tensor of shape
                           (batch_size, 1, seq_length, seq_length).

        Returns:
            Tensor: Output tensor with logits.
        """
        x = self.encoder(x, mask)
        x = self.projection(x)
        return x  # Logits are used directly for BCEWithLogitsLoss

    def training_step(self, batch: VesselBatch, batch_idx: int) -> Tensor:
        """
        Training step where the loss is computed.

        Args:
            batch (dict): A dictionary containing input tensors and targets.
            batch_idx (int): Index of the batch.

        Returns:
            Tensor: Computed loss value.
        """

        logits = self(batch.data, batch.mask)
        loss = self.loss_fn(logits.squeeze(), batch.labels)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch: VesselBatch, batch_idx: int) -> Tensor:
        """
        Validation step where the loss is computed.

        Args:
            batch (dict): A dictionary containing input tensors and targets.
            batch_idx (int): Index of the batch.

        Returns:
            Tensor: Computed loss value.
        """
        logits = self(batch.data, batch.mask)
        loss = self.loss_fn(logits.squeeze(), batch.labels)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> optim.Optimizer:
        """
        Configure the optimizer.

        Returns:
            optim.Optimizer: The Adam optimizer.
        """
        return optim.Adam(self.parameters(), lr=self.config.learning_rate)
