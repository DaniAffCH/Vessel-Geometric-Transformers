import lightning as L
import torch
from torch import Tensor, nn, optim
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score

from config.dataclasses import GatrConfig
from src.data.datamodule import VesselBatch
from src.lib.geometricAlgebraElements import GeometricAlgebraBase
from src.models.layers.geometric import EquiLinearLayer, GATrBlock


class Gatr(L.LightningModule):  # type: ignore[misc]
    """
    Geometric Algebra Transformer model for classification.

    Args:
        config (GatrConfig): Configuration object
                                 for setting model parameters.
    """

    def __init__(self, config: GatrConfig) -> None:
        super(Gatr, self).__init__()

        self.hsProjection = EquiLinearLayer(
            config.features_size_limit * 4, config.hidden_size
        )

        self.backbone = nn.ModuleList(
            [
                GATrBlock(config.hidden_size, config.num_attention_heads)
                for _ in range(config.num_backbone_layers)
            ]
        )

        self.outputProjection = EquiLinearLayer(config.hidden_size, 1)

        self.finalProjection = nn.Linear(GeometricAlgebraBase.GA_size, 1)

        self.config = config
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Initialize metrics
        self.train_accuracy = BinaryAccuracy()
        self.val_accuracy = BinaryAccuracy()
        self.test_accuracy = BinaryAccuracy()

        self.train_f1 = BinaryF1Score()
        self.val_f1 = BinaryF1Score()
        self.test_f1 = BinaryF1Score()

        # Enabling fp16 precision to increase performance in Daniele's GPU ;)
        torch.set_float32_matmul_precision("medium")

    def getReference(self, x: Tensor) -> torch.Tensor:
        # shape (batch, 1, ..., 1, 16)
        dim = tuple(range(1, len(x.shape) - 1))
        return torch.mean(x, dim=dim, keepdim=True)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Forward pass through the model.

        Args:
            x (Tensor): Input tensor of shape
                        (batch_size, num_elements, seq_length, ga_size).
            mask (Tensor): Mask tensor of shape
                           (batch_size, num_elements, seq_length).

        Returns:
            Tensor: Output tensor with logits.
        """
        # (batch_size, num_elements*seq_length, ga_size)
        x = x.reshape(x.size(0), -1, x.size(-1))

        # (batch_size, num_elements*seq_length)
        mask = mask.reshape(x.size(0), -1)

        reference = self.getReference(x)

        x = self.hsProjection(x)

        for layer in self.backbone:
            x = layer(x, reference)

        x = self.outputProjection(x)
        x = x.reshape(x.shape[0], -1)
        x = self.finalProjection(x)

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
        logits = logits.squeeze(-1)
        loss = self.loss_fn(logits, batch.labels)

        preds = logits.sigmoid() > 0.5  # Convert logits to binary predictions
        self.train_accuracy(preds, batch.labels)
        self.train_f1(preds, batch.labels)

        self.log(
            "train/loss", loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/acc",
            self.train_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/f1",
            self.train_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
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
        logits = logits.squeeze(-1)
        loss = self.loss_fn(logits, batch.labels)

        preds = logits.sigmoid() > 0.5
        self.val_accuracy(preds, batch.labels)
        self.val_f1(preds, batch.labels)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val/acc",
            self.val_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True
        )

        return loss

    def test_step(self, batch: VesselBatch, batch_idx: int) -> Tensor:
        """
        Test step where the loss is computed.

        Args:
            batch (dict): A dictionary containing input tensors and targets.
            batch_idx (int): Index of the batch.

        Returns:
            Tensor: Computed loss value.
        """
        logits = self(batch.data, batch.mask)
        logits = logits.squeeze(-1)

        loss = self.loss_fn(logits, batch.labels)

        preds = logits.sigmoid() > 0.5
        self.test_accuracy(preds, batch.labels)
        self.test_f1(preds, batch.labels)

        self.log(
            "test/loss", loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "test/acc",
            self.test_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test/f1",
            self.test_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def configure_optimizers(self) -> optim.Optimizer:
        """
        Configure the optimizer.

        Returns:
            optim.Optimizer: The Adam optimizer.
        """
        return optim.Adam(self.parameters(), lr=self.config.learning_rate)
