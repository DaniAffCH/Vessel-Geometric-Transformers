import yaml
from omegaconf import OmegaConf

from config import Config


def load_config(file_path: str) -> Config:
    with open(file_path, "r") as file:
        try:
            config: Config = OmegaConf.structured(Config)
            data = OmegaConf.create(yaml.safe_load(file))
            OmegaConf.unsafe_merge(config, data)
            return config
        except yaml.YAMLError as e:
            print(f"Error decoding YAML: {e}")
            return Config()
