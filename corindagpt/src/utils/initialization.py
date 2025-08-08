from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_config_path() -> Path:
    """Return the default configuration file path within the project tree."""
    return PROJECT_ROOT / "config" / "config.yaml"


# Load environment variables from a local .env if present (no override of existing env)
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load and parse the YAML configuration file, with environment variable overrides.

    Environment variables take precedence if set:
    - OPENAI_API_KEY
    - ELEVENLABS_API_KEY
    - MODEL_NAME_TEXT (optional)
    - MODEL_NAME_EMBEDDING (optional)
    - MODEL_NAME_TRANSCRIPTION (optional)

    Args:
        config_path: Optional explicit path to the config YAML. If not provided,
            uses the canonical project path corindagpt/config/config.yaml.

    Returns:
        A dictionary with configuration values.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
    """
    path = Path(config_path) if config_path is not None else _default_config_path()

    if not path.exists():
        raise FileNotFoundError(f"Config file not found at: {path}")

    with path.open("r", encoding="utf-8") as file:
        data: Dict[str, Any] = yaml.safe_load(file) or {}

    # Apply environment variable overrides
    openai_from_env = os.getenv("OPENAI_API_KEY")
    eleven_from_env = os.getenv("ELEVENLABS_API_KEY")

    if openai_from_env:
        data["openai_api_key"] = openai_from_env
    if eleven_from_env:
        data["elevenlabs_api_key"] = eleven_from_env

    model_text_from_env = os.getenv("MODEL_NAME_TEXT")
    model_embed_from_env = os.getenv("MODEL_NAME_EMBEDDING")
    model_transcribe_from_env = os.getenv("MODEL_NAME_TRANSCRIPTION")

    if model_text_from_env or model_embed_from_env or model_transcribe_from_env:
        model_names: Dict[str, Any] = data.get("model_names") or {}
        if model_text_from_env:
            model_names["text"] = model_text_from_env
        if model_embed_from_env:
            model_names["embedding"] = model_embed_from_env
        if model_transcribe_from_env:
            model_names["transcription"] = model_transcribe_from_env
        data["model_names"] = model_names

    logger.debug(
        "Loaded configuration from %s with keys: %s (env overrides applied: %s)",
        path,
        list(data.keys()),
        [
            k
            for k in [
                "OPENAI_API_KEY",
                "ELEVENLABS_API_KEY",
                "MODEL_NAME_TEXT",
                "MODEL_NAME_EMBEDDING",
                "MODEL_NAME_TRANSCRIPTION",
            ]
            if os.getenv(k)
        ],
    )
    return data
