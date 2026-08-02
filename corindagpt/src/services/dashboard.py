"""Localhost settings dashboard.

A tiny stdlib HTTP server (no new dependencies) that lets the performer edit
config/settings.yaml from a browser and have supported values apply to the
running show immediately - no restart. Runs as a daemon thread inside the
Effie process, bound to 127.0.0.1 only.

The page is generated from SETTINGS_SCHEMA, so adding a future setting to
the schema (and to the apply callback, if it needs live wiring) is all it
takes for it to appear in the browser.
"""
from __future__ import annotations

import html
import logging
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import parse_qs

import yaml

logger = logging.getLogger(__name__)


# Each entry: key (settings.yaml name), label, help text, and a validator
# spec. kind 'int' uses lo/hi clamping; kind 'choice' snaps to choices[0]
# when the submitted value is unrecognized. All v1 settings apply live.
SETTINGS_SCHEMA: List[Dict[str, Any]] = [
    {
        "key": "phases",
        "label": "Performance phases",
        "help": "1 = simple mode (persona only). 2-5 use the phase prompt files.",
        "kind": "int",
        "lo": 1,
        "hi": 5,
        "default": 1,
    },
    {
        "key": "response_playback",
        "label": "Answer playback",
        "help": "queued = answers wait for the pull string (doll mode). immediate = answers play as soon as ready.",
        "kind": "choice",
        "choices": ["queued", "immediate"],
        "default": "queued",
    },
    {
        "key": "keepalive_interval_s",
        "label": "API keepalive (seconds)",
        "help": "Heartbeat that keeps OpenAI/ElevenLabs connections hot between questions. 0 disables.",
        "kind": "int",
        "lo": 0,
        "hi": 3600,
        "default": 60,
    },
]


def validate_setting(spec: Dict[str, Any], raw: Any) -> Any:
    """Clamp/snap a submitted value to something the show can always run with."""
    if spec["kind"] == "int":
        try:
            value = int(str(raw).strip())
        except Exception:
            return spec["default"]
        return min(spec["hi"], max(spec["lo"], value))
    if spec["kind"] == "choice":
        value = str(raw).strip().lower()
        return value if value in spec["choices"] else spec["choices"][0]
    return spec["default"]


def read_settings(settings_path: Path) -> Dict[str, Any]:
    try:
        with settings_path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return {}
    except Exception as exc:
        logger.warning("Dashboard: settings unreadable (%s)", exc)
        return {}


def write_settings(settings_path: Path, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Merge updates into settings.yaml atomically, preserving unknown keys.

    Comments in the file are not preserved (PyYAML round-trip limitation);
    the file header comment block is re-emitted so the file stays
    self-documenting.
    """
    current = read_settings(settings_path)
    current.update(updates)
    header = (
        "# Effie show settings\n"
        "#\n"
        "# Performer-tunable values. Edited by the localhost dashboard while\n"
        "# the show runs; also safe to edit by hand. Technical configuration\n"
        "# lives in config.yaml - settings here override it at startup.\n"
    )
    body = yaml.safe_dump(current, default_flow_style=False, sort_keys=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(settings_path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(header + body)
        os.replace(tmp_path, settings_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return current


def _render_page(settings_path: Path, saved: bool) -> str:
    current = read_settings(settings_path)
    rows = []
    for spec in SETTINGS_SCHEMA:
        key = spec["key"]
        value = current.get(key, spec["default"])
        label = html.escape(spec["label"])
        help_text = html.escape(spec["help"])
        if spec["kind"] == "choice":
            options = "".join(
                f'<option value="{html.escape(c)}"{" selected" if c == value else ""}>{html.escape(c)}</option>'
                for c in spec["choices"]
            )
            field = f'<select name="{key}">{options}</select>'
        else:
            field = (
                f'<input type="number" name="{key}" value="{html.escape(str(value))}" '
                f'min="{spec["lo"]}" max="{spec["hi"]}">'
            )
        rows.append(
            f'<div class="row"><label>{label}</label>{field}'
            f'<p class="help">{help_text}</p></div>'
        )
    notice = '<p class="saved">Saved - live settings applied.</p>' if saved else ""
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Effie settings</title>
<style>
  body {{ font-family: Georgia, serif; background: #1a1418; color: #e8e0d8;
         max-width: 34rem; margin: 3rem auto; padding: 0 1rem; }}
  h1 {{ font-weight: normal; letter-spacing: 0.08em; }}
  .row {{ margin: 1.4rem 0; }}
  label {{ display: block; margin-bottom: 0.3rem; }}
  input, select {{ font-size: 1rem; padding: 0.3rem 0.5rem; background: #2a2228;
                   color: #e8e0d8; border: 1px solid #5a4a55; border-radius: 4px; }}
  .help {{ font-size: 0.82rem; color: #9a8a92; margin: 0.3rem 0 0; }}
  .saved {{ color: #9ac79a; }}
  button {{ font-size: 1rem; padding: 0.5rem 1.6rem; background: #5a4a55;
            color: #e8e0d8; border: none; border-radius: 4px; cursor: pointer; }}
</style></head>
<body>
<h1>Effie &mdash; show settings</h1>
{notice}
<form method="post" action="/">
{''.join(rows)}
<button type="submit">Save</button>
</form>
</body></html>"""


def _make_handler(settings_path: Path, apply_cb: Optional[Callable[[Dict[str, Any]], None]]):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # route to logging, not stderr
            logger.debug("Dashboard: " + fmt, *args)

        def _send_html(self, content: str, status: int = 200) -> None:
            data = content.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            saved = "saved=1" in (self.path.split("?", 1)[1] if "?" in self.path else "")
            self._send_html(_render_page(settings_path, saved))

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            form = parse_qs(self.rfile.read(length).decode("utf-8"))
            updates: Dict[str, Any] = {}
            for spec in SETTINGS_SCHEMA:
                if spec["key"] in form:
                    updates[spec["key"]] = validate_setting(spec, form[spec["key"]][0])
            try:
                write_settings(settings_path, updates)
            except Exception as exc:
                logger.error("Dashboard: failed writing settings: %s", exc)
                self._send_html("<p>Save failed - see log.</p>", status=500)
                return
            if apply_cb is not None:
                try:
                    apply_cb(updates)
                except Exception as exc:
                    logger.error("Dashboard: live-apply failed (file saved): %s", exc)
            logger.info("Dashboard: settings saved and applied: %s", updates)
            self.send_response(303)
            self.send_header("Location", "/?saved=1")
            self.end_headers()

    return Handler


def start_dashboard(
    config: Dict[str, Any],
    settings_path: Path,
    apply_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Optional[ThreadingHTTPServer]:
    """Start the dashboard thread if enabled; returns the server or None.

    Binds to 127.0.0.1 only - the dashboard is for the machine Effie runs on,
    never the network.
    """
    dash_cfg = config.get("dashboard") or {}
    if not bool(dash_cfg.get("enabled", False)):
        return None
    port = int(dash_cfg.get("port", 7360))
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), _make_handler(settings_path, apply_cb))
    except OSError as exc:
        logger.warning("Dashboard: could not bind 127.0.0.1:%s (%s); dashboard off", port, exc)
        return None
    thread = threading.Thread(target=server.serve_forever, name="dashboard", daemon=True)
    thread.start()
    logger.info("Dashboard: http://127.0.0.1:%s", port)
    return server
