import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from config.dataclasses import TrainerConfig


# TODO metter wandb qui
class VesselTrainer(L.Trainer):  # type: ignore[misc]
    def __init__(self, config: TrainerConfig):

        self.config = config

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=config.ckpt_path,
            save_last=True,  # Keep track of the model at the last epoch
            verbose=True,
        )

        early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            patience=config.patience,
            verbose=True,
        )

        self.trainer = L.Trainer(
            max_epochs=config.max_epochs,
            callbacks=[checkpoint_callback, early_stopping_callback],
        )

    def fit(
        self, model: L.LightningModule, datamodule: L.LightningDataModule
    ) -> None:
        self.trainer.fit(
            model=model,
            train_dataloaders=datamodule.train_dataloader(),
            val_dataloaders=datamodule.val_dataloader(),
            ckpt_path="last",
        )  # TODO mettere il path giusto per riprender il checkpoint
