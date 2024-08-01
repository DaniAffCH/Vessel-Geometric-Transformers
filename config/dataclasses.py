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


@dataclass
class TrainerConfig:
    max_epochs: int = MISSING
    patience: int = MISSING
    ckpt_path: str = MISSING


@dataclass
class BaselineConfig:
    learning_rate: float = MISSING


@dataclass
class GatrConfig:
    learning_rate: float = MISSING


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    gatr: GatrConfig = field(default_factory=GatrConfig)
