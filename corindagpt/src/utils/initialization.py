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


def _default_settings_path() -> Path:
    """Performer-tunable show settings; overrides config.yaml at load time."""
    return PROJECT_ROOT / "config" / "settings.yaml"


def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_merge(dst[key], value)
        else:
            dst[key] = value


def _apply_show_settings(data: Dict[str, Any], settings_path: Path) -> None:
    """Overlay show settings (settings.yaml) onto the loaded config, in place.

    Settings are performer-facing knobs (the future dashboard writes that file
    only); invalid values are clamped with a warning rather than raised, so a
    bad edit can't keep the show from starting.
    """
    if not settings_path.exists():
        return
    try:
        with settings_path.open("r", encoding="utf-8") as fh:
            settings: Dict[str, Any] = yaml.safe_load(fh) or {}
    except Exception as exc:
        logger.warning("Show settings unreadable (%s); using config.yaml as-is.", exc)
        return
    data["settings"] = settings

    # Dashboard-written overrides of arbitrary config.yaml values live under
    # a nested `config:` subtree; merge them over the loaded defaults first
    overrides = settings.get("config")
    if isinstance(overrides, dict):
        _deep_merge(data, overrides)

    if "phases" in settings:
        try:
            phases = int(settings.get("phases"))
        except Exception:
            phases = 1
        if not 1 <= phases <= 5:
            logger.warning("settings.yaml: phases=%r out of range; clamping to 1-5.", settings.get("phases"))
            phases = min(5, max(1, phases))
        data["performance_plan"] = list(range(1, phases + 1))

    if "response_playback" in settings:
        mode = str(settings.get("response_playback")).strip().lower()
        if mode not in ("queued", "immediate"):
            logger.warning("settings.yaml: response_playback=%r invalid; using 'queued'.", settings.get("response_playback"))
            mode = "queued"
        data.setdefault("audio_queue", {})["response_playback"] = mode

    if "keepalive_interval_s" in settings:
        try:
            interval = float(settings.get("keepalive_interval_s"))
        except Exception:
            interval = 60.0
        if not 0 <= interval <= 3600:
            logger.warning("settings.yaml: keepalive_interval_s=%r out of range; clamping to 0-3600.", settings.get("keepalive_interval_s"))
            interval = min(3600.0, max(0.0, interval))
        data.setdefault("network", {})["keepalive_interval_s"] = interval


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

    # Overlay performer-facing show settings (settings.yaml lives beside the
    # config; explicit config_path is used by tests, which keep their own data)
    if config_path is None:
        _apply_show_settings(data, _default_settings_path())

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
