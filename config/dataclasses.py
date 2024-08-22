from dataclasses import dataclass, field

from omegaconf import MISSING


@dataclass
class DatasetConfig:
    drive_url: str = MISSING
    name: str = MISSING
    download_path: str = MISSING
    bifurcating_path: str = MISSING
    single_path: str = MISSING
    train_size: float = MISSING
    val_size: float = MISSING
    test_size: float = MISSING
    batch_size: int = MISSING
    features_size_limit: int = MISSING


@dataclass
class TrainerConfig:
    max_epochs: int = MISSING
    patience: int = MISSING
    ckpt_path: str = MISSING
    wandb_project: str = MISSING
    min_delta: float = MISSING
    resume_training: bool = MISSING


@dataclass
class BaselineConfig:
    transformer_embedding_dim: int = MISSING
    transformer_num_heads: int = MISSING
    transformer_feedforward_dim: int = MISSING
    transformer_num_layers: int = MISSING
    learning_rate: float = MISSING
    features_size_limit: int = MISSING


@dataclass
class GatrConfig:
    hidden_size: int = MISSING
    learning_rate: float = MISSING
    features_size_limit: int = MISSING


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    gatr: GatrConfig = field(default_factory=GatrConfig)
