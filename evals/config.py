"""
Configuration loader and validator.

Loads YAML config files and validates required fields.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigError(Exception):
    """Raised when config is invalid or missing required fields."""
    pass


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and validate configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Validated config dictionary

    Raises:
        ConfigError: If config is invalid or missing required fields
    """
    path = Path(config_path)

    if not path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    try:
        with open(path) as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse YAML: {e}")

    if config is None:
        raise ConfigError(f"Config file is empty: {config_path}")

    # Validate required top-level sections
    _validate_config(config)

    return config


def _validate_config(config: Dict[str, Any]) -> None:
    """
    Validate config has all required fields.

    Args:
        config: Config dictionary to validate

    Raises:
        ConfigError: If required fields are missing
    """
    required_sections = ["endpoint", "model", "sampling", "run"]

    for section in required_sections:
        if section not in config:
            raise ConfigError(f"Missing required section: {section}")

    # Validate endpoint section
    endpoint = config["endpoint"]
    if "base_url" not in endpoint:
        raise ConfigError("Missing endpoint.base_url")

    # Validate run section
    run = config["run"]
    required_run_fields = ["dataset", "output_dir"]
    for field in required_run_fields:
        if field not in run:
            raise ConfigError(f"Missing run.{field}")

    # Validate sampling section
    sampling = config["sampling"]
    if "temperature" not in sampling:
        raise ConfigError("Missing sampling.temperature")
    if "max_tokens" not in sampling:
        raise ConfigError("Missing sampling.max_tokens")


def get_endpoint_url(config: Dict[str, Any]) -> str:
    """Extract endpoint base URL from config."""
    return config["endpoint"]["base_url"]


def get_api_key(config: Dict[str, Any]) -> Optional[str]:
    """Extract API key from config (may be None)."""
    return config["endpoint"].get("api_key")


def get_model_name(config: Dict[str, Any]) -> str:
    """Extract model name from config."""
    return config["model"]


def get_dataset_path(config: Dict[str, Any]) -> str:
    """Extract dataset path from config."""
    return config["run"]["dataset"]


def get_output_dir(config: Dict[str, Any]) -> str:
    """Extract output directory from config."""
    return config["run"]["output_dir"]


def get_run_name(config: Dict[str, Any]) -> Optional[str]:
    """Extract run name from config (may be None)."""
    return config["run"].get("run_name")


def get_parallelism(config: Dict[str, Any]) -> int:
    """Extract parallelism level from config (default: 1)."""
    return config["run"].get("parallelism", 1)


def get_seed(config: Dict[str, Any]) -> Optional[int]:
    """Extract seed from config (may be None)."""
    return config["run"].get("seed")


def get_sampling_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract all sampling parameters."""
    return config["sampling"]
