import os

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import wandb
from config.dataclasses import TrainerConfig


class VesselTrainer(L.Trainer):  # type: ignore[misc]
    def __init__(self, config: TrainerConfig):

        self.config = config

        wandb.init(  # type: ignore[attr-defined]
            project=config.wandb_project,
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="val/loss",
            dirpath=config.ckpt_path,
            save_last=True,  # Keep track of the model at the last epoch
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
            callbacks=[checkpoint_callback, early_stopping_callback],
            logger=WandbLogger(),
        )

    def fit(
        self, model: L.LightningModule, datamodule: L.LightningDataModule
    ) -> None:
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
        super().test(
            model=model,
            dataloaders=datamodule.test_dataloader(),
            ckpt_path=f"{self.config.ckpt_path}/last.ckpt",
        )
