import lightning as L
import optuna

from config.dataclasses import Config
from src.data.datamodule import VesselDataModule
from src.models import BaselineTransformer
from src.models.gatr import Gatr


def optuna_callback(study: optuna.study.Study, trial: optuna.Trial) -> None:
    print(f"Trial {trial.number} finished with value {trial.value}")


def baseline_hpo(config: Config) -> None:

    def objective(trial: optuna.Trial) -> float:

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
        model = BaselineTransformer(config.baseline)
        data = VesselDataModule(config.dataset)

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
        lambda trial: objective(trial),
        n_trials=config.optuna.n_trials,
        callbacks=[optuna_callback],
    )

    config.baseline.learning_rate = study.best_params["lr"]
    config.baseline.transformer_num_heads = study.best_params["num_heads"]
    config.baseline.transformer_num_layers = study.best_params["num_layers"]
    config.dataset.batch_size = study.best_params["batch_size"]


def gatr_hpo(config: Config) -> None:

    def objective(trial: optuna.Trial) -> float:
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
        model = Gatr(config.gatr)
        data = VesselDataModule(config.dataset)

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
        lambda trial: objective(trial),
        n_trials=config.optuna.n_trials,
        callbacks=[optuna_callback],
    )

    config.gatr.learning_rate = study.best_params["lr"]
    config.gatr.num_attention_heads = study.best_params["num_heads"]
    config.gatr.num_backbone_layers = study.best_params["num_layers"]
    config.dataset.batch_size = study.best_params["batch_size"]
