"""
Configuration loader for Car Price Prediction project.
Loads settings from config.yaml with environment variable overrides.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Singleton configuration manager."""

    _instance: Optional["Config"] = None
    _config: Dict[str, Any] = {}

    def __new__(cls) -> "Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not self._config:
            self._load_config()

    def _load_config(self, config_path: Optional[Path] = None) -> None:
        """Load configuration from YAML file."""
        if config_path is None:
            # Find config.yaml from project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config.yaml"

        with open(config_path, "r") as f:
            self._config = yaml.safe_load(f)

        # Override with environment variables (prefix: CAR_PRICE_)
        import os
        for key, value in os.environ.items():
            if key.startswith("CAR_PRICE_"):
                config_key = key[len("CAR_PRICE_"):].lower()
                self._set_nested(config_key, value)

    def _set_nested(self, key: str, value: str) -> None:
        """Set nested config value using dot notation (e.g., 'model.xgboost.n_estimators')."""
        keys = key.split(".")
        d = self._config
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        # Try to convert to appropriate type
        d[keys[-1]] = self._convert_type(value)

    def _convert_type(self, value: str) -> Any:
        """Convert string to appropriate Python type."""
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value using dot notation."""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    @property
    def all(self) -> Dict[str, Any]:
        return self._config.copy()


# Global config instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config