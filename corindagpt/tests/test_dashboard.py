from __future__ import annotations

from pathlib import Path

import yaml

from src.services import dashboard
from src.utils.initialization import _apply_show_settings


def _spec(key: str):
    return next(s for s in dashboard.SETTINGS_SCHEMA if s["key"] == key)


def test_validate_int_clamps_and_defaults():
    phases = _spec("phases")
    assert dashboard.validate_setting(phases, "3") == 3
    assert dashboard.validate_setting(phases, "99") == 5
    assert dashboard.validate_setting(phases, "-2") == 1
    assert dashboard.validate_setting(phases, "not a number") == 1


def test_validate_choice_snaps_to_default():
    playback = _spec("response_playback")
    assert dashboard.validate_setting(playback, "immediate") == "immediate"
    assert dashboard.validate_setting(playback, "QUEUED") == "queued"
    assert dashboard.validate_setting(playback, "loudly") == "queued"


def test_validate_float_and_bool():
    stability = _spec("tts.elevenlabs.voice_settings.stability")
    assert dashboard.validate_setting(stability, "0.55") == 0.55
    assert dashboard.validate_setting(stability, "7") == 1.0
    assert dashboard.validate_setting(stability, "junk") == 0.40
    boost = _spec("tts.elevenlabs.voice_settings.use_speaker_boost")
    assert dashboard.validate_setting(boost, "on") is True
    assert dashboard.validate_setting(boost, "") is False


def test_write_settings_top_level_and_nested(tmp_path: Path):
    path = tmp_path / "settings.yaml"
    path.write_text("phases: 2\nfuture_setting: keep-me\n", encoding="utf-8")
    dashboard.write_settings(
        path,
        {
            "phases": 4,
            "tts.elevenlabs.voice_settings.speed": 1.0,
            "input_patterns.hotkey": "f12",
        },
    )
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["phases"] == 4
    assert data["future_setting"] == "keep-me"
    assert data["config"]["tts"]["elevenlabs"]["voice_settings"]["speed"] == 1.0
    assert data["config"]["input_patterns"]["hotkey"] == "f12"


def test_settings_config_subtree_overlays_config(tmp_path: Path):
    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text(
        "config:\n  tts:\n    elevenlabs:\n      voice_settings:\n        speed: 1.15\n",
        encoding="utf-8",
    )
    data = {"tts": {"elevenlabs": {"voice_settings": {"speed": 0.9, "stability": 0.4}}}}
    _apply_show_settings(data, settings_file)
    assert data["tts"]["elevenlabs"]["voice_settings"]["speed"] == 1.15
    assert data["tts"]["elevenlabs"]["voice_settings"]["stability"] == 0.4  # untouched


def test_current_value_precedence(tmp_path: Path):
    spec = _spec("tts.elevenlabs.voice_settings.speed")
    settings = {"config": {"tts": {"elevenlabs": {"voice_settings": {"speed": 1.2}}}}}
    effective = {"tts": {"elevenlabs": {"voice_settings": {"speed": 0.9}}}}
    assert dashboard.current_value(spec, settings, effective) == 1.2  # override wins
    assert dashboard.current_value(spec, {}, effective) == 0.9  # effective config
    assert dashboard.current_value(spec, {}, {}) == 0.90  # schema default


def test_listening_mode_in_schema_and_page(tmp_path: Path):
    spec = _spec("listening_mode")
    assert spec["live"] is True
    assert dashboard.validate_setting(spec, "streaming") == "streaming"
    assert dashboard.validate_setting(spec, "nonsense") == "push_hold"
    path = tmp_path / "settings.yaml"
    path.write_text("listening_mode: streaming\n", encoding="utf-8")
    page = dashboard._render_page(path, saved=False, restart=False)
    assert '<option value="streaming" selected>' in page
    assert 'data-tab="Live"' in page
    assert 'live.json' in page


def test_render_page_has_tabs_and_badges(tmp_path: Path):
    path = tmp_path / "settings.yaml"
    path.write_text("phases: 3\nresponse_playback: immediate\n", encoding="utf-8")
    page = dashboard._render_page(path, saved=False, restart=False)
    for tab in dashboard.TABS:
        assert f'data-tab="{tab}"' in page
    assert 'name="phases" value="3"' in page
    assert '<option value="immediate" selected>' in page
    assert "restart required" in page  # badge on non-live settings
    assert 'name="tts.elevenlabs.voice_settings.stability"' in page


def test_start_dashboard_disabled_returns_none(tmp_path: Path):
    assert dashboard.start_dashboard({}, tmp_path / "settings.yaml") is None
    assert dashboard.start_dashboard({"dashboard": {"enabled": False}}, tmp_path / "settings.yaml") is None
