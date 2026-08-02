"""Localhost settings dashboard.

A tiny stdlib HTTP server (no new dependencies) that lets the performer edit
config/settings.yaml from a browser. Runs as a daemon thread inside the
Effie process, bound to 127.0.0.1 only.

The page is generated from SETTINGS_SCHEMA (tabbed by category). The
dashboard writes ONLY settings.yaml - never config.yaml, which remains the
pristine defaults file. Top-level performer keys (phases, response_playback,
keepalive_interval_s) keep their existing shape; everything else is stored
under a nested `config:` subtree that initialization deep-merges over
config.yaml at load time.

Live vs restart: values the app reads per question/answer (marked
live=True) take effect immediately; the rest are constructed at startup and
the page badges them "restart required".
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

TABS = ["Show", "Voice", "Controls", "Listening", "System"]

# Each entry:
#   key       - form name and settings.yaml location. top=True keys are
#               top-level performer settings; top=False keys are dot paths
#               into the `config:` override subtree (mirroring config.yaml).
#   display   - dot path into the EFFECTIVE merged config for showing the
#               current value when settings.yaml has no override yet.
#   kind      - int | float | choice | bool | str (validated/clamped on save)
#   live      - True if the running show picks the change up without restart.
SETTINGS_SCHEMA: List[Dict[str, Any]] = [
    # ------------------------------------------------------------- Show
    {
        "key": "phases", "top": True, "tab": "Show",
        "label": "Performance phases",
        "help": "1 = simple mode (persona only). 2-5 use the phase prompt files.",
        "kind": "int", "lo": 1, "hi": 5, "default": 1, "live": True,
        "display": None,  # special-cased: len(performance_plan)
    },
    {
        "key": "response_playback", "top": True, "tab": "Show",
        "label": "Answer playback",
        "help": "queued = answers wait for the pull string (doll mode). immediate = answers play as soon as ready.",
        "kind": "choice", "choices": ["queued", "immediate"], "default": "queued", "live": True,
        "display": "audio_queue.response_playback",
    },
    {
        "key": "audio_queue.preload.order", "top": False, "tab": "Show",
        "label": "Preloaded clip order",
        "help": "Order the 11 ambient doll phrases play in.",
        "kind": "choice", "choices": ["sequential", "random"], "default": "sequential", "live": False,
        "display": "audio_queue.preload.order",
    },
    {
        "key": "audio_queue.preload.loop", "top": False, "tab": "Show",
        "label": "Loop preloaded clips",
        "help": "Played clips return to the back of the queue.",
        "kind": "bool", "default": True, "live": False,
        "display": "audio_queue.preload.loop",
    },
    {
        "key": "memory.max_turns", "top": False, "tab": "Show",
        "label": "Memory (conversation turns)",
        "help": "How many exchanges Effie remembers for context.",
        "kind": "int", "lo": 0, "hi": 50, "default": 10, "live": False,
        "display": "memory.max_turns",
    },
    # ------------------------------------------------------------ Voice
    {
        "key": "tts.elevenlabs.voice_id", "top": False, "tab": "Voice",
        "label": "ElevenLabs voice ID",
        "help": "The voice Effie speaks with.",
        "kind": "str", "default": "", "live": True,
        "display": "tts.elevenlabs.voice_id",
    },
    {
        "key": "tts.elevenlabs.model_id", "top": False, "tab": "Voice",
        "label": "TTS model",
        "help": "eleven_flash_v2_5 is the low-latency choice (~200 ms).",
        "kind": "str", "default": "eleven_flash_v2_5", "live": True,
        "display": "tts.elevenlabs.model_id",
    },
    {
        "key": "tts.elevenlabs.voice_settings.stability", "top": False, "tab": "Voice",
        "label": "Stability",
        "help": "Lower = more expressive variation, higher = more consistent.",
        "kind": "float", "lo": 0.0, "hi": 1.0, "default": 0.40, "live": True,
        "display": "tts.elevenlabs.voice_settings.stability",
    },
    {
        "key": "tts.elevenlabs.voice_settings.speed", "top": False, "tab": "Voice",
        "label": "Speed",
        "help": "1.0 = normal. Effie currently speaks slightly slow (0.90).",
        "kind": "float", "lo": 0.5, "hi": 1.5, "default": 0.90, "live": True,
        "display": "tts.elevenlabs.voice_settings.speed",
    },
    {
        "key": "tts.elevenlabs.voice_settings.similarity_boost", "top": False, "tab": "Voice",
        "label": "Similarity boost",
        "help": "How closely generation sticks to the cloned voice.",
        "kind": "float", "lo": 0.0, "hi": 1.0, "default": 0.40, "live": True,
        "display": "tts.elevenlabs.voice_settings.similarity_boost",
    },
    {
        "key": "tts.elevenlabs.voice_settings.style", "top": False, "tab": "Voice",
        "label": "Style exaggeration",
        "help": "0 = neutral delivery; higher costs latency.",
        "kind": "float", "lo": 0.0, "hi": 1.0, "default": 0.0, "live": True,
        "display": "tts.elevenlabs.voice_settings.style",
    },
    {
        "key": "tts.elevenlabs.voice_settings.use_speaker_boost", "top": False, "tab": "Voice",
        "label": "Speaker boost",
        "help": "Extra similarity processing; adds a little latency.",
        "kind": "bool", "default": False, "live": True,
        "display": "tts.elevenlabs.voice_settings.use_speaker_boost",
    },
    # --------------------------------------------------------- Controls
    {
        "key": "input_patterns.hotkey", "top": False, "tab": "Controls",
        "label": "Record trigger key",
        "help": "Hold = decoded question, double-tap+hold = bypass decoder.",
        "kind": "str", "default": "`", "live": False,
        "display": "input_patterns.hotkey",
    },
    {
        "key": "input_patterns.play_hotkey", "top": False, "tab": "Controls",
        "label": "Play trigger key",
        "help": "Tap = play queue front (the pull string). Empty folds playback onto the record key's tap.",
        "kind": "str", "default": "space", "live": False,
        "display": "input_patterns.play_hotkey",
    },
    {
        "key": "input_patterns.brief_max_ms", "top": False, "tab": "Controls",
        "label": "Tap max (ms)",
        "help": "Presses up to this long count as a tap.",
        "kind": "int", "lo": 50, "hi": 1000, "default": 250, "live": False,
        "display": "input_patterns.brief_max_ms",
    },
    {
        "key": "input_patterns.sustained_min_ms", "top": False, "tab": "Controls",
        "label": "Hold min (ms)",
        "help": "Presses at least this long count as a hold (recording).",
        "kind": "int", "lo": 200, "hi": 3000, "default": 600, "live": False,
        "display": "input_patterns.sustained_min_ms",
    },
    {
        "key": "input_patterns.compound_double_press_window_ms", "top": False, "tab": "Controls",
        "label": "Double-tap window (ms)",
        "help": "A second press inside this window makes a double-tap+hold.",
        "kind": "int", "lo": 0, "hi": 1000, "default": 350, "live": False,
        "display": "input_patterns.compound_double_press_window_ms",
    },
    {
        "key": "transitions.phase_transition.hotkey", "top": False, "tab": "Controls",
        "label": "Phase advance key",
        "help": "Long-press advances to the next phase.",
        "kind": "str", "default": "f11", "live": False,
        "display": "transitions.phase_transition.hotkey",
    },
    {
        "key": "transitions.phase_transition.long_press_ms", "top": False, "tab": "Controls",
        "label": "Phase advance hold (ms)",
        "help": "How long the phase key must be held.",
        "kind": "int", "lo": 500, "hi": 10000, "default": 3000, "live": False,
        "display": "transitions.phase_transition.long_press_ms",
    },
    # -------------------------------------------------------- Listening
    {
        "key": "transcription.elevenlabs.model", "top": False, "tab": "Listening",
        "label": "Transcription model",
        "help": "ElevenLabs speech-to-text model.",
        "kind": "str", "default": "scribe_v1", "live": False,
        "display": "transcription.elevenlabs.model",
    },
    {
        "key": "transcription.language", "top": False, "tab": "Listening",
        "label": "Language",
        "help": "Spoken language hint (e.g. en).",
        "kind": "str", "default": "en", "live": False,
        "display": "transcription.language",
    },
    {
        "key": "transcription.prompt", "top": False, "tab": "Listening",
        "label": "Transcription hint",
        "help": "Phrases the transcriber should expect - helps with unusual words.",
        "kind": "str", "default": "", "live": False,
        "display": "transcription.prompt",
    },
    {
        "key": "transcription.elevenlabs.diarize", "top": False, "tab": "Listening",
        "label": "Diarization",
        "help": "Speaker separation. Off for a single close mic - it only adds latency.",
        "kind": "bool", "default": False, "live": False,
        "display": "transcription.elevenlabs.diarize",
    },
    {
        "key": "transcription.elevenlabs.tag_audio_events", "top": False, "tab": "Listening",
        "label": "Audio event tags",
        "help": "Tags like (laughter) in transcripts. Off - the decoder ignores them.",
        "kind": "bool", "default": False, "live": False,
        "display": "transcription.elevenlabs.tag_audio_events",
    },
    # ----------------------------------------------------------- System
    {
        "key": "model_names.text", "top": False, "tab": "System",
        "label": "LLM model",
        "help": "OpenAI model that generates Effie's answers.",
        "kind": "str", "default": "gpt-5.4-nano", "live": True,
        "display": "model_names.text",
    },
    {
        "key": "transitions.llm_phase_control.enabled", "top": False, "tab": "System",
        "label": "Spoken phase control",
        "help": "Let phase keyphrases (e.g. 'next phase') switch phases. Only active in multi-phase shows.",
        "kind": "bool", "default": True, "live": False,
        "display": "transitions.llm_phase_control.enabled",
    },
    {
        "key": "keepalive_interval_s", "top": True, "tab": "System",
        "label": "API keepalive (seconds)",
        "help": "Heartbeat that keeps OpenAI/ElevenLabs connections hot between questions. 0 disables.",
        "kind": "int", "lo": 0, "hi": 3600, "default": 60, "live": True,
        "display": "network.keepalive_interval_s",
    },
    {
        "key": "dashboard.port", "top": False, "tab": "System",
        "label": "Dashboard port",
        "help": "Where this page is served (127.0.0.1 only).",
        "kind": "int", "lo": 1024, "hi": 65535, "default": 7360, "live": False,
        "display": "dashboard.port",
    },
]


def get_by_path(data: Dict[str, Any], dotted: str) -> Any:
    node: Any = data
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def set_by_path(data: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = data
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value


def validate_setting(spec: Dict[str, Any], raw: Any) -> Any:
    """Clamp/snap a submitted value to something the show can always run with."""
    kind = spec["kind"]
    if kind == "int":
        try:
            value = int(str(raw).strip())
        except Exception:
            return spec["default"]
        return min(spec["hi"], max(spec["lo"], value))
    if kind == "float":
        try:
            value = float(str(raw).strip())
        except Exception:
            return spec["default"]
        return round(min(spec["hi"], max(spec["lo"], value)), 3)
    if kind == "choice":
        value = str(raw).strip().lower()
        return value if value in spec["choices"] else spec["choices"][0]
    if kind == "bool":
        return str(raw).strip().lower() in ("on", "true", "1", "yes")
    if kind == "str":
        return str(raw).strip()[:300]
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
    """Merge validated updates into settings.yaml atomically.

    `updates` maps schema keys to values; top=True keys land at the top
    level, others nest under the `config:` override subtree. Unknown
    existing keys are preserved. Comments are not preserved (PyYAML), so a
    header block is re-emitted to keep the file self-documenting.
    """
    specs = {s["key"]: s for s in SETTINGS_SCHEMA}
    current = read_settings(settings_path)
    for key, value in updates.items():
        spec = specs.get(key)
        if spec is None:
            continue
        if spec["top"]:
            current[key] = value
        else:
            overrides = current.setdefault("config", {})
            set_by_path(overrides, key, value)
    header = (
        "# Effie show settings\n"
        "#\n"
        "# Performer-tunable values. Edited by the localhost dashboard while\n"
        "# the show runs; also safe to edit by hand. Technical configuration\n"
        "# lives in config.yaml; the `config:` block here overrides it.\n"
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


def _effective_config() -> Dict[str, Any]:
    try:
        from ..utils.initialization import load_config

        return load_config()
    except Exception as exc:
        logger.warning("Dashboard: could not load effective config (%s)", exc)
        return {}


def current_value(spec: Dict[str, Any], settings: Dict[str, Any], effective: Dict[str, Any]) -> Any:
    """Settings.yaml override wins; otherwise the effective merged config."""
    if spec["top"]:
        if spec["key"] in settings:
            return settings[spec["key"]]
    else:
        override = get_by_path(settings.get("config") or {}, spec["key"])
        if override is not None:
            return override
    if spec["key"] == "phases":
        plan = effective.get("performance_plan")
        return len(plan) if isinstance(plan, list) and plan else spec["default"]
    if spec.get("display"):
        value = get_by_path(effective, spec["display"])
        if value is not None:
            return value
    return spec["default"]


def _render_field(spec: Dict[str, Any], value: Any) -> str:
    key = spec["key"]
    label = html.escape(spec["label"])
    help_text = html.escape(spec["help"])
    badge = "" if spec["live"] else ' <span class="badge">restart required</span>'
    kind = spec["kind"]
    if kind == "choice":
        options = "".join(
            f'<option value="{html.escape(c)}"{" selected" if c == value else ""}>{html.escape(c)}</option>'
            for c in spec["choices"]
        )
        field = f'<select name="{key}">{options}</select>'
    elif kind == "bool":
        checked = " checked" if bool(value) else ""
        field = f'<input type="checkbox" name="{key}"{checked}>'
    elif kind == "float":
        field = (
            f'<input type="number" name="{key}" value="{html.escape(str(value))}" '
            f'min="{spec["lo"]}" max="{spec["hi"]}" step="0.05">'
        )
    elif kind == "int":
        field = (
            f'<input type="number" name="{key}" value="{html.escape(str(value))}" '
            f'min="{spec["lo"]}" max="{spec["hi"]}">'
        )
    else:
        field = f'<input type="text" name="{key}" value="{html.escape(str(value))}" size="40">'
    return (
        f'<div class="row"><label>{label}{badge}</label>{field}'
        f'<p class="help">{help_text}</p></div>'
    )


def _render_page(settings_path: Path, saved: bool, restart: bool) -> str:
    settings = read_settings(settings_path)
    effective = _effective_config()
    sections = []
    nav = []
    for i, tab in enumerate(TABS):
        rows = "".join(
            _render_field(spec, current_value(spec, settings, effective))
            for spec in SETTINGS_SCHEMA
            if spec["tab"] == tab
        )
        active = " active" if i == 0 else ""
        nav.append(f'<button type="button" class="tab{active}" data-tab="{tab}">{tab}</button>')
        sections.append(f'<section id="tab-{tab}" class="panel{active}">{rows}</section>')
    notice = ""
    if saved:
        extra = " Restart Effie for the badged changes to take effect." if restart else ""
        notice = f'<p class="saved">Saved - live settings applied.{extra}</p>'
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Effie settings</title>
<style>
  body {{ font-family: Georgia, serif; background: #1a1418; color: #e8e0d8;
         max-width: 38rem; margin: 2.5rem auto; padding: 0 1rem; }}
  h1 {{ font-weight: normal; letter-spacing: 0.08em; }}
  nav {{ display: flex; gap: 0.4rem; margin: 1.2rem 0; flex-wrap: wrap; }}
  .tab {{ font-size: 0.95rem; padding: 0.45rem 1rem; background: #2a2228; color: #9a8a92;
          border: 1px solid #3a2f37; border-radius: 6px 6px 0 0; cursor: pointer; }}
  .tab.active {{ background: #5a4a55; color: #e8e0d8; }}
  .panel {{ display: none; }}
  .panel.active {{ display: block; }}
  .row {{ margin: 1.3rem 0; }}
  label {{ display: block; margin-bottom: 0.3rem; }}
  input, select {{ font-size: 1rem; padding: 0.3rem 0.5rem; background: #2a2228;
                   color: #e8e0d8; border: 1px solid #5a4a55; border-radius: 4px; }}
  input[type=checkbox] {{ width: 1.1rem; height: 1.1rem; }}
  .help {{ font-size: 0.82rem; color: #9a8a92; margin: 0.3rem 0 0; }}
  .badge {{ font-size: 0.7rem; color: #c7a26a; border: 1px solid #c7a26a;
            border-radius: 3px; padding: 0.05rem 0.35rem; margin-left: 0.5rem;
            vertical-align: middle; }}
  .saved {{ color: #9ac79a; }}
  .actions {{ margin: 2rem 0; }}
  button[type=submit] {{ font-size: 1rem; padding: 0.5rem 1.8rem; background: #5a4a55;
            color: #e8e0d8; border: none; border-radius: 4px; cursor: pointer; }}
</style></head>
<body>
<h1>Effie &mdash; show settings</h1>
{notice}
<form method="post" action="/">
<nav>{''.join(nav)}</nav>
{''.join(sections)}
<div class="actions"><button type="submit">Save all</button></div>
</form>
<script>
document.querySelectorAll('.tab').forEach(btn => btn.addEventListener('click', () => {{
  document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
}}));
</script>
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
            query = self.path.split("?", 1)[1] if "?" in self.path else ""
            self._send_html(
                _render_page(settings_path, saved="saved=1" in query, restart="restart=1" in query)
            )

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            form = parse_qs(self.rfile.read(length).decode("utf-8"))
            settings = read_settings(settings_path)
            effective = _effective_config()
            updates: Dict[str, Any] = {}
            restart_needed = False
            for spec in SETTINGS_SCHEMA:
                key = spec["key"]
                if spec["kind"] == "bool":
                    # Single form: all fields always submit, so an absent
                    # checkbox genuinely means unchecked
                    value = validate_setting(spec, form.get(key, [""])[0])
                elif key in form:
                    value = validate_setting(spec, form[key][0])
                else:
                    continue
                before = current_value(spec, settings, effective)
                updates[key] = value
                if not spec["live"] and value != before:
                    restart_needed = True
            try:
                write_settings(settings_path, updates)
            except Exception as exc:
                logger.error("Dashboard: failed writing settings: %s", exc)
                self._send_html("<p>Save failed - see log.</p>", status=500)
                return
            if apply_cb is not None:
                try:
                    apply_cb({k: v for k, v in updates.items() if k in ("phases", "response_playback", "keepalive_interval_s")})
                except Exception as exc:
                    logger.error("Dashboard: live-apply failed (file saved): %s", exc)
            logger.info("Dashboard: settings saved (%d values, restart_needed=%s)", len(updates), restart_needed)
            self.send_response(303)
            self.send_header("Location", "/?saved=1" + ("&restart=1" if restart_needed else ""))
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
