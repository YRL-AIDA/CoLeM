import json
import yaml
from typing import Any


class Config:
    """Wrapper configuration class.

    Args:
        config_path: Configuration filename.
    """
    def __init__(self, config_path: str = "config.yaml"):
        self._open_yaml(config_path)

    def _open_json(self, config_path: str) -> None:
        """Parse config in `json` format."""
        with open(config_path, mode="r", encoding="utf-8") as f:
            self.data = json.load(f)

    def _open_yaml(self, config_path: str) -> None:
        """Parse config in `yaml` format."""
        with open(config_path, mode="r") as f:
            self.data = yaml.safe_load(f)

    def __getitem__(self, item: str):
        return self.data[item]

    def get(self, item: str, default_value: Any) -> Any:
        """Get item from data dictionary."""
        return self.data.get(item, default_value)
