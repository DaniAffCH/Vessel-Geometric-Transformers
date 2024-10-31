from abc import ABC, abstractmethod
from typing import List

import lightning as L
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch import Tensor, nn, optim
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score

from src.data.datamodule import VesselBatch


class VesselClassificationModel(L.LightningModule, ABC):  # type: ignore[misc]
    """
    Base model for binary classification of vessels. This class contains all
    what is needed to train a model for vessel classification. It defines the
    metrics, and the training, validation and test steps, which are common to
    all possible models. The only two things which need to be defined by the
    concrete implementations are:
    - The forward method, which defines the model architecture.
    - The configure_optimizers method, which defines the optimizer.
    """

    def __init__(self) -> None:
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Initialize metrics
        self.train_accuracy = BinaryAccuracy()
        self.val_accuracy = BinaryAccuracy()
        self.test_accuracy = BinaryAccuracy()

        self.train_f1 = BinaryF1Score()
        self.val_f1 = BinaryF1Score()
        self.test_f1 = BinaryF1Score()

        self.test_y: List[torch.Tensor] = []
        self.test_y_hat: List[torch.Tensor] = []

        # Enabling fp16 precision to increase performance in Daniele's GPU ;)
        torch.set_float32_matmul_precision("medium")

    @abstractmethod
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Forward pass through the model. The mask is needed for the transformer.

        Args:
            x (Tensor): Input tensor of shape
                        (batch_size, num_elements, seq_length, ga_size).
            mask (Tensor): Mask tensor of shape
                           (batch_size, num_elements, seq_length).

        Returns:
            Tensor: Output tensor with logits.
        """
        pass

    @abstractmethod
    def configure_optimizers(self) -> optim.Optimizer:
        """
        Configure the optimizer.

        Returns:
            optim.Optimizer: An optimizer object.
        """
        pass

    def training_step(self, batch: VesselBatch, batch_idx: int) -> Tensor:
        """
        Training step where the loss is computed.

        Args:
            batch (dict): A dictionary containing input tensors and targets.
            batch_idx (int): Index of the batch.

        Returns:
            Tensor: Computed loss value.
        """
        logits: torch.Tensor = self(batch.data, batch.mask)
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
        logits: torch.Tensor = self(batch.data, batch.mask)
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
        logits: torch.Tensor = self(batch.data, batch.mask)
        logits = logits.squeeze(-1)

        loss = self.loss_fn(logits, batch.labels)

        preds = logits.sigmoid() > 0.5
        self.test_accuracy(preds, batch.labels)
        self.test_f1(preds, batch.labels)

        self.test_y_hat.append(preds)
        self.test_y.append(batch.labels)

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

    def on_test_epoch_end(self) -> None:
        y_hat = torch.cat(self.test_y_hat)
        y = torch.cat(self.test_y)

        cm = confusion_matrix(y.cpu(), y_hat.cpu())

        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix - Test Set")
        plt.show()
