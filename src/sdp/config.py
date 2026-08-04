"""Configuration loading.

Configuration is data, not code. Keeping it in YAML means an experiment can be
changed without editing Python, and the exact configuration of any run can be
recorded alongside its results.
"""

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "app.yaml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load a YAML configuration file."""
    config_path = path or DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
