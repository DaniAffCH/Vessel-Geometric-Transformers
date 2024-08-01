import yaml
from omegaconf import OmegaConf

from config import Config


def load_config(file_path: str) -> Config:
    """
    Load configuration from a YAML file and merge it into a Config object.

    Args:
        file_path (str): The path to the YAML configuration file.

    Returns:
        Config: The merged configuration object.
    """
    with open(file_path, "r") as file:
        try:
            config: Config = OmegaConf.structured(Config)
            data = OmegaConf.create(yaml.safe_load(file))
            OmegaConf.unsafe_merge(config, data)
            return config
        except yaml.YAMLError as e:
            print(f"Error decoding YAML: {e}")
            return Config()
