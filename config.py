"""TODO"""
import json
from typing import Any


class Config:
    """Wrapper configuration class.

    Args:
        config_path: Configuration filename.
    """
    def __init__(self, config_path: str = "config.json"):
        with open(config_path, mode="r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __getitem__(self, item: str):
        return self.data[item]

    def get(self, item: str, default_value: Any):
        """Get item from data dictionary."""
        return self.data.get(item, default_value)


if __name__ == "__main__":
    c = Config()
