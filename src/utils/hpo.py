import lightning as L
import optuna

from config.dataclasses import Config
from src.models.baseline import BaselineTransformer
from src.models.gatr import Gatr


def optuna_callback(study: optuna.study.Study, trial: optuna.Trial) -> None:
    print(f"Trial {trial.number} finished with value {trial.value}")


def baseline_hpo(config: Config, data: L.LightningDataModule) -> None:
    """
    Perform hyperparameter optimization for the Baseline Transformer model
    using Optuna.
    Args:
        config (Config): Configuration object containing hyperparameter
        settings and other configurations.
        data (L.LightningDataModule): LightningDataModule object containing
        the dataset.
    Returns:
        None
    This function initializes an Optuna study to optimize hyperparameters
    for the Baseline Transformer model. It defines an objective function
    that suggests hyperparameters, trains the model, and evaluates its
    performance. The best hyperparameters found during the optimization
    are then updated in the provided configuration object.
    The hyperparameters optimized include:
        - Learning rate
        - Number of attention heads
        - Number of transformer layers
        - Batch size
    In the end, the optimal parameters are written into the config object.
    """

    print("Starting a new hyperparameter optimization study...")

    def objective(
        trial: optuna.Trial,
        model: L.LightningModule,
        data: L.LightningDataModule,
    ) -> float:

        print("Starting a new trial...")
        print(f"Trial number: {trial.number}")

        # Hyperparameters to optimize
        config.baseline.learning_rate = trial.suggest_float(
            "lr", 1e-4, 1e-1, log=True
        )
        config.baseline.transformer_num_heads = trial.suggest_categorical(
            "num_heads", choices=[2, 4, 8]
        )
        config.baseline.transformer_num_layers = trial.suggest_categorical(
            "num_layers", choices=[1, 2, 3]
        )
        config.dataset.batch_size = trial.suggest_categorical(
            "batch_size", choices=[2, 4, 8, 16]
        )

        print(f"Learning rate: {config.baseline.learning_rate}")
        print(f"Attention Heads: {config.baseline.transformer_num_heads}")
        print(f"Attention Layers: {config.baseline.transformer_num_layers}")
        print(f"Batch Size: {config.dataset.batch_size}")

        trainer = L.Trainer(max_epochs=3)

        trainer.fit(model, data)
        optimized_value: float = trainer.logged_metrics["val/loss"]
        return optimized_value

    model = BaselineTransformer(config.baseline)
    sampler = optuna.samplers.TPESampler(seed=config.optuna.seed)
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name="baseline_hpo",
    )

    study.optimize(
        lambda trial: objective(trial, model, data),
        n_trials=config.optuna.n_trials,
        callbacks=[optuna_callback],
    )

    print("Hyperparameter optimization completed.")
    print(
        f"Best hyperparameters found:\n\
          Learning rate: {study.best_params['lr']}\n\
          Number of Attention Heads: {study.best_params['num_heads']}\n\
          Number of Attention Layers: {study.best_params['num_layers']}\n\
          Batch Size: {study.best_params['batch_size']}\n"
    )

    config.baseline.learning_rate = study.best_params["lr"]
    config.baseline.transformer_num_heads = study.best_params["num_heads"]
    config.baseline.transformer_num_layers = study.best_params["num_layers"]
    config.dataset.batch_size = study.best_params["batch_size"]


def gatr_hpo(config: Config, data: L.LightningDataModule) -> None:
    """
    Perform hyperparameter optimization for the GATR model using Optuna.
    Args:
        config (Config): Configuration object containing hyperparameter
        settings and other configurations.
        data (L.LightningDataModule): LightningDataModule object containing
        the dataset.
    Returns:
        None
    This function initializes an Optuna study to optimize hyperparameters
    for the GATR model. It defines an objective function that suggests
    hyperparameters, trains the model, and evaluates its performance. The
    best hyperparameters found during the optimization are then updated in
    the provided configuration object.
    The hyperparameters optimized include:
        - Learning rate
        - Number of attention heads
        - Number of backbone layers
        - Batch size
    """

    print("Starting a new hyperparameter optimization study...")

    def objective(
        trial: optuna.Trial,
        model: L.LightningModule,
        data: L.LightningDataModule,
    ) -> float:
        print("Starting a new trial...")
        print(f"Trial number: {trial.number}")

        # Hyperparameters to optimize
        config.gatr.learning_rate = trial.suggest_float(
            "lr", 1e-4, 1e-1, log=True
        )
        config.gatr.num_attention_heads = trial.suggest_categorical(
            "num_heads", choices=[2, 4, 8]
        )
        config.gatr.num_backbone_layers = trial.suggest_categorical(
            "num_layers", choices=[1, 2, 3]
        )
        config.dataset.batch_size = trial.suggest_categorical(
            "batch_size", choices=[2, 4, 8, 16]
        )

        print(f"Learning rate: {config.gatr.learning_rate}")
        print(f"Attention Heads: {config.gatr.num_attention_heads}")
        print(f"Attention Layers: {config.gatr.num_backbone_layers}")
        print(f"Batch Size: {config.dataset.batch_size}")

        trainer = L.Trainer(max_epochs=2)

        trainer.fit(model, data)
        optimized_value: float = trainer.logged_metrics["val/loss"]
        return optimized_value

    model = Gatr(config.gatr)
    sampler = optuna.samplers.TPESampler(seed=config.optuna.seed)
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name="gatr_hpo",
    )

    study.optimize(
        lambda trial: objective(trial, model, data),
        n_trials=config.optuna.n_trials,
        callbacks=[optuna_callback],
    )

    print("Hyperparameter optimization completed.")
    print(
        f"Best hyperparameters found:\n\
          Learning rate: {study.best_params['lr']}\n\
          Number of Attention Heads: {study.best_params['num_heads']}\n\
          Number of Attention Layers: {study.best_params['num_layers']}\n\
          Batch Size: {study.best_params['batch_size']}\n"
    )

    config.gatr.learning_rate = study.best_params["lr"]
    config.gatr.num_attention_heads = study.best_params["num_heads"]
    config.gatr.num_backbone_layers = study.best_params["num_layers"]
    config.dataset.batch_size = study.best_params["batch_size"]
