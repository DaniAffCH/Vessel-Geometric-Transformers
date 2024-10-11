import os

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import wandb
from config.dataclasses import TrainerConfig


class VesselTrainer(L.Trainer):  # type: ignore[misc]
    """
    This class extends the lightning Trainer to add
    gatr specific configurations
    """

    def __init__(self, config: TrainerConfig):

        self.config = config

        wandb.login(key=config.wandb_api_key)  # type: ignore
        wandb.init(  # type: ignore
            project=config.wandb_project,
        )

        self._checkpoint_callback = ModelCheckpoint(
            monitor="val/loss",
            dirpath=config.ckpt_path,
            save_top_k=1,
            mode="min",
            verbose=True,
        )

        early_stopping_callback = EarlyStopping(
            monitor="val/loss",
            patience=config.patience,
            min_delta=config.min_delta,
            verbose=True,
        )

        super().__init__(
            max_epochs=config.max_epochs,
            callbacks=[self._checkpoint_callback, early_stopping_callback],
            logger=WandbLogger(),
        )

    def fit(
        self, model: L.LightningModule, datamodule: L.LightningDataModule
    ) -> None:
        """Train the model, resuming a previous checkpoint if needed."""

        checkpoint_path = f"{self.config.ckpt_path}/last.ckpt"

        if self.config.resume_training and os.path.exists(checkpoint_path):
            ckpt_path = checkpoint_path
        else:
            ckpt_path = None

        super().fit(
            model=model,
            train_dataloaders=datamodule.train_dataloader(),
            val_dataloaders=datamodule.val_dataloader(),
            ckpt_path=ckpt_path,
        )

    def test(
        self, model: L.LightningModule, datamodule: L.LightningDataModule
    ) -> None:
        """Test the model on the test set."""
        best_ckpt_path = self._checkpoint_callback.best_model_path

        if not best_ckpt_path:
            raise ValueError(
                "Best model path not found. \
                Make sure the model was trained and saved."
            )

        super().test(
            model=model,
            dataloaders=datamodule.test_dataloader(),
            ckpt_path=best_ckpt_path,
        )
