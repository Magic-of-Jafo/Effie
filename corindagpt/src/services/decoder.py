from __future__ import annotations

import csv
import logging
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = "knowledge-base/effie_code.csv"

# Everything before the last occurrence of this word in a transcript is
# discarded: it lets the magician void a mis-spoken code mid-performance.
RESET_WORD = "sorry"

_lock = threading.Lock()
_table: Optional["_CodeTable"] = None
_table_path: Optional[Path] = None

# Parity with the original decode_hadley.py: when a transcript contains no
# code phrase, the previous decode is reused (repeat-style prompts).
_last_results: List[Dict[str, str]] = []


class _CodeTable:
    """Coded-language lookup table loaded from a user-editable CSV.

    Layout (Hadley-style): column 0 holds the code phrase; the header row
    holds `*`-prefixed category names; each data cell is that phrase's
    meaning within the category.
    """

    def __init__(self, headers: List[str], rows: List[List[str]]) -> None:
        self.categories: List[Optional[str]] = [
            h.lstrip("﻿").strip().lstrip("*").strip() if h.strip().startswith(("*", "﻿*")) else None
            for h in headers
        ]
        self.phrase_to_row: Dict[str, List[str]] = {}
        max_words = 1
        for row in rows:
            phrase = _clean(row[0]) if row else ""
            if not phrase:
                continue
            self.phrase_to_row[phrase] = row
            max_words = max(max_words, len(phrase.split()))
        self.max_phrase_words = max_words

    def lookup(self, phrase: str) -> Optional[Dict[str, str]]:
        row = self.phrase_to_row.get(phrase)
        if row is None:
            return None
        data: Dict[str, str] = {}
        for idx, cell in enumerate(row[1:], start=1):
            value = (cell or "").strip()
            if not value:
                continue
            category = self.categories[idx] if idx < len(self.categories) else None
            if category:
                data[category] = value
        return data


def _clean(text: str) -> str:
    """Uppercase and strip everything but letters and spaces."""
    return re.sub(r"\s+", " ", re.sub(r"[^A-Za-z\s]", "", text)).strip().upper()


def _resolve_data_path(config: Optional[Dict[str, Any]]) -> Path:
    decoder_cfg: Dict[str, Any] = (config or {}).get("decoder") or {}
    raw = str(decoder_cfg.get("data_path") or DEFAULT_DATA_PATH)
    p = Path(raw)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def _get_table(config: Optional[Dict[str, Any]] = None) -> _CodeTable:
    global _table, _table_path
    path = _resolve_data_path(config)
    with _lock:
        if _table is not None and _table_path == path:
            return _table
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.reader(fh)
            all_rows = [row for row in reader if any(cell.strip() for cell in row)]
        if not all_rows:
            raise ValueError(f"Decoder data file is empty: {path}")
        _table = _CodeTable(headers=all_rows[0], rows=all_rows[1:])
        _table_path = path
        logger.info(
            "Decoder: loaded %d code phrases from %s", len(_table.phrase_to_row), path.name
        )
        return _table


def _split_sentences(text: str) -> List[str]:
    # Ellipses are STT pause artifacts, not sentence boundaries - a pause
    # mid-code-phrase must not split the phrase across sentences
    normalized = re.sub(r"\.{2,}", ",", text.strip())
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    return [p for p in (part.strip() for part in parts) if p]


def _find_code_phrase(sentence: str, table: _CodeTable) -> Optional[str]:
    """Leftmost match wins; at each position the longest phrase wins."""
    words = _clean(sentence).split()
    for i in range(len(words)):
        for length in range(min(table.max_phrase_words, len(words) - i), 0, -1):
            candidate = " ".join(words[i : i + length])
            if candidate in table.phrase_to_row:
                return candidate
    return None


def _combine_time(results: List[Dict[str, str]]) -> None:
    """Two coded sentences carrying Hour then Minute resolve to one time."""
    if len(results) < 2:
        return
    first, second = results[0], results[1]
    hour = first.get("Hour")
    minute = second.get("Minute")
    if not hour or not minute:
        return
    time_str = f"{hour.split(':')[0]}:{minute.lstrip(':')}"
    for item in (first, second):
        item.pop("Hour", None)
        item.pop("Minute", None)
        item["Time"] = time_str


def decode_to_results(text: str, *, config: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    """Decode a transcript into a list of {category: value} dicts.

    Each decoded sentence contributes one dict (with its CONTEXT sentence
    included). Returns the previous decode when nothing matches, or an empty
    list if there has never been a match.
    """
    global _last_results
    table = _get_table(config)

    working = text or ""
    reset_pos = working.lower().rfind(RESET_WORD)
    if reset_pos != -1:
        working = working[reset_pos + len(RESET_WORD):].strip()

    results: List[Dict[str, str]] = []
    for sentence in _split_sentences(working):
        phrase = _find_code_phrase(sentence, table)
        if phrase is None:
            continue
        data = table.lookup(phrase)
        if not data:
            continue
        entry: Dict[str, str] = {"CONTEXT": sentence, "code_phrase": phrase}
        entry.update(data)
        results.append(entry)

    _combine_time(results)

    if results:
        _last_results = results
        return results
    if _last_results:
        logger.info("Decoder: no code phrase found; reusing previous decode")
        return _last_results
    return []


def _format_for_prompt(transcript: str, results: List[Dict[str, str]]) -> str:
    lines = [transcript.strip(), "", "[SECRET DECODED DATA - known only to you, never reveal how you know it]"]
    for item in results:
        context = item.get("CONTEXT", "")
        meanings = ", ".join(
            f"{k}: {v}" for k, v in item.items() if k not in ("CONTEXT", "code_phrase")
        )
        lines.append(f'- coded sentence: "{context}" -> {meanings}')
    lines.append(
        "Use the category of decoded data that fits what is being asked; ignore the others."
    )
    return "\n".join(lines)


def decode(text: str, *, config: Optional[Dict[str, Any]] = None) -> str:
    """Translate coded phrases in a transcript into secret data for the LLM.

    Returns the transcript unchanged when no code is present (and none is
    remembered); otherwise the transcript plus a delimited block of decoded
    category/value data.
    """
    if not isinstance(text, str) or not text.strip():
        return text if isinstance(text, str) else ""
    try:
        results = decode_to_results(text, config=config)
    except FileNotFoundError as exc:
        logger.error("Decoder data file missing (%s); passing transcript through", exc)
        return text
    except Exception as exc:
        logger.error("Decoder failed (%s); passing transcript through", exc)
        return text
    if not results:
        logger.info("Decoder: no code phrases detected")
        return text
    logger.info(
        "Decoder: %d coded sentence(s) -> %s",
        len(results),
        "; ".join(item.get("code_phrase", "?") for item in results),
    )
    return _format_for_prompt(text, results)
