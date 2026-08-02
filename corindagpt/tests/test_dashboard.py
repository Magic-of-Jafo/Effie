from __future__ import annotations

from pathlib import Path

import yaml

from src.services import dashboard


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


def test_write_settings_merges_and_preserves_unknown_keys(tmp_path: Path):
    path = tmp_path / "settings.yaml"
    path.write_text("phases: 2\nfuture_setting: keep-me\n", encoding="utf-8")
    dashboard.write_settings(path, {"phases": 4, "keepalive_interval_s": 0})
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["phases"] == 4
    assert data["keepalive_interval_s"] == 0
    assert data["future_setting"] == "keep-me"


def test_write_settings_creates_missing_file(tmp_path: Path):
    path = tmp_path / "settings.yaml"
    dashboard.write_settings(path, {"phases": 1})
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == {"phases": 1}


def test_render_page_shows_current_values(tmp_path: Path):
    path = tmp_path / "settings.yaml"
    path.write_text("phases: 3\nresponse_playback: immediate\n", encoding="utf-8")
    page = dashboard._render_page(path, saved=False)
    assert 'name="phases" value="3"' in page
    assert '<option value="immediate" selected>' in page
    assert 'name="keepalive_interval_s"' in page  # default renders when absent


def test_start_dashboard_disabled_returns_none(tmp_path: Path):
    assert dashboard.start_dashboard({}, tmp_path / "settings.yaml") is None
    assert dashboard.start_dashboard({"dashboard": {"enabled": False}}, tmp_path / "settings.yaml") is None
