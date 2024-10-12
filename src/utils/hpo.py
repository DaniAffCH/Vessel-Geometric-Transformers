import lightning as L
import optuna

from config.dataclasses import Config


def optuna_callback(study: optuna.study.Study, trial: optuna.Trial) -> None:
    print(f"Trial {trial.number} finished with value {trial.value}")


def baseline_hpo(
    config: Config, model: L.LightningModule, data: L.LightningDataModule
) -> None:

    print("Starting a new hyperparameter optimization study...")

    def objective(
        trial: optuna.Trial,
        model: L.LightningModule,
        data: L.LightningDataModule,
    ) -> float:

        # Hyperparameters to optimize
        config.baseline.learning_rate = trial.suggest_float(
            "lr", 1e-5, 1e-1, log=True
        )
        config.baseline.transformer_num_heads = trial.suggest_int(
            "num_heads", 2, 4, step=2
        )
        config.baseline.transformer_num_layers = trial.suggest_int(
            "num_layers", 1, 3
        )
        config.dataset.batch_size = trial.suggest_int("batch_size", 2, 5)

        trainer = L.Trainer(max_epochs=3)

        trainer.fit(model, data)
        optimized_value: float = trainer.logged_metrics["val/loss"]
        return optimized_value

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


def gatr_hpo(
    config: Config, model: L.LightningModule, data: L.LightningDataModule
) -> None:

    print("Starting a new hyperparameter optimization study...")

    def objective(
        trial: optuna.Trial,
        model: L.LightningModule,
        data: L.LightningDataModule,
    ) -> float:
        # Hyperparameters to optimize
        config.gatr.learning_rate = trial.suggest_float(
            "lr", 1e-5, 1e-1, log=True
        )
        config.gatr.num_attention_heads = trial.suggest_int(
            "num_heads", 2, 4, step=2
        )
        config.gatr.num_backbone_layers = trial.suggest_int("num_layers", 1, 3)
        config.dataset.batch_size = trial.suggest_int("batch_size", 2, 5)

        trainer = L.Trainer(max_epochs=2)

        trainer.fit(model, data)
        optimized_value: float = trainer.logged_metrics["val/loss"]
        return optimized_value

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
