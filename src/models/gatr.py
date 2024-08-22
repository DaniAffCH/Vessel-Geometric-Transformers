import lightning as L
import torch
from torch import Tensor, nn, optim
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score

from config.dataclasses import GatrConfig
from src.data.datamodule import VesselBatch
from src.models.layers.geometric.equiLinear import EquiLinearLayer


class Gatr(L.LightningModule):  # type: ignore[misc]
    """
    Geometric Algebra Transformer model for classification.

    Args:
        config (GatrConfig): Configuration object
                                 for setting model parameters.
    """

    def __init__(self, config: GatrConfig) -> None:
        super().__init__()

        self.equilinear = EquiLinearLayer(
            config.features_size_limit * 4, config.hidden_size
        )

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

        print(x.shape)
        x = self.equilinear(x)
        print(x.shape)

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
