from attr import dataclass
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


@dataclass
class Config:
    dataset: DatasetConfig = DatasetConfig()
