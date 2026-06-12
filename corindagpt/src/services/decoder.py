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


def _is_header_cell(cell: str) -> bool:
    return (cell or "").lstrip("﻿").strip().startswith("*")


def _header_name(cell: str) -> str:
    return (cell or "").lstrip("﻿").strip().lstrip("*").strip()


class _CodeTable:
    """Coded-language lookup table loaded from a user-editable CSV.

    Layout (Hadley-style): column 0 holds the code phrase; `*`-prefixed
    cells are category headers. Headers may appear mid-table - a column's
    category for a given row is the nearest `*` cell ABOVE it (e.g. column
    13 is *Month for rows 1-29 and becomes *Vegetable from row 31 on).
    This mirrors the upward scan in the original project's get_key().
    """

    def __init__(self, grid: List[List[str]]) -> None:
        n_cols = max(len(row) for row in grid)
        # Walk top to bottom tracking the active header per column, so each
        # data row knows its categories at the time it appears
        active: List[Optional[str]] = [None] * n_cols
        self._row_categories: Dict[int, List[Optional[str]]] = {}
        self.phrase_to_rowidx: Dict[str, int] = {}
        self._rows: Dict[int, List[str]] = {}
        max_words = 1
        for r, row in enumerate(grid):
            # Categories visible to this row come from rows strictly above
            self._row_categories[r] = list(active)
            for c in range(n_cols):
                cell = row[c] if c < len(row) else ""
                if _is_header_cell(cell):
                    active[c] = _header_name(cell)
            phrase = _clean(row[0]) if row and not _is_header_cell(row[0]) else ""
            if phrase:
                self.phrase_to_rowidx[phrase] = r
                self._rows[r] = row
                max_words = max(max_words, len(phrase.split()))
        self.max_phrase_words = max_words

    @property
    def phrase_to_row(self) -> Dict[str, int]:  # membership checks in matching
        return self.phrase_to_rowidx

    def lookup(self, phrase: str) -> Optional[Dict[str, str]]:
        r = self.phrase_to_rowidx.get(phrase)
        if r is None:
            return None
        row = self._rows[r]
        categories = self._row_categories[r]
        data: Dict[str, str] = {}
        for idx in range(1, len(row)):
            value = (row[idx] or "").strip()
            if not value or _is_header_cell(value):
                continue
            category = categories[idx] if idx < len(categories) else None
            if category:
                data[category] = value
        return data


# STT spelling variants that must match the code table's wording
# (ALRIGHT/ALLRIGHT handling carried over from the original decode_en_rapport.py)
_WORD_NORMALIZATIONS = {
    "ALRIGHT": "ALL RIGHT",
    "ALLRIGHT": "ALL RIGHT",
    "OK": "OKAY",
}


def _clean(text: str) -> str:
    """Uppercase, strip everything but letters/spaces, normalize STT variants."""
    cleaned = re.sub(r"\s+", " ", re.sub(r"[^A-Za-z\s]", "", text)).strip().upper()
    words = [_WORD_NORMALIZATIONS.get(w, w) for w in cleaned.split()]
    return " ".join(words)


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
        _table = _CodeTable(all_rows)
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


def _find_codes(sentences: List[str], table: _CodeTable) -> List[Tuple[str, str]]:
    """Find code phrases across the whole transcript as one word stream.

    STT inserts punctuation by guesswork ("Cool. Could you..."), so phrases
    must be allowed to span sentence boundaries. Leftmost match wins and the
    longest phrase wins at each position. After a match, scanning resumes at
    the next sentence: later code-words in the same sentence (e.g. the TELL
    in "could you tell me...") are conversational, not codes.

    Returns a list of (phrase, context) pairs in transcript order.
    """
    words: List[Tuple[str, int]] = []  # (cleaned word, sentence index)
    for si, sentence in enumerate(sentences):
        for w in _clean(sentence).split():
            words.append((w, si))

    found: List[Tuple[str, str]] = []
    i, n = 0, len(words)
    while i < n:
        matched = None
        for length in range(min(table.max_phrase_words, n - i), 0, -1):
            candidate = " ".join(w for w, _ in words[i : i + length])
            if candidate in table.phrase_to_row:
                matched = (candidate, length)
                break
        if matched is None:
            i += 1
            continue
        phrase, length = matched
        start_sent = words[i][1]
        end_sent = words[i + length - 1][1]
        context = " ".join(sentences[start_sent : end_sent + 1])
        found.append((phrase, context))
        # Resume at the first word of a later sentence
        i += length
        while i < n and words[i][1] <= end_sent:
            i += 1
    return found


_TIME_ASK_RE = re.compile(r"\b(time|clock|watch|o'?clock)\b", re.IGNORECASE)


def _apply_single_code_rule(results: List[Dict[str, str]], transcript: str) -> List[Dict[str, str]]:
    """One code per input - never offer the model competing rows.

    The sole legitimate two-code pattern is the time ask: the first code
    carries an Hour and the second a Minute, which collapse into a single
    entry holding only the combined Time. Hour/Minute are columns on nearly
    every row, so the data alone can't tell a time ask from an accidental
    second code - the question has to actually mention time/clock/watch.
    All other extra detections are accidental (a later sentence happening
    to open with a code word) and are dropped.
    """
    if len(results) < 2:
        return results
    first, second = results[0], results[1]
    hour = first.get("Hour")
    minute = second.get("Minute")
    if hour and minute and _TIME_ASK_RE.search(transcript or ""):
        return [
            {
                "CONTEXT": f"{first.get('CONTEXT', '')} {second.get('CONTEXT', '')}".strip(),
                "code_phrase": f"{first.get('code_phrase', '')} + {second.get('code_phrase', '')}",
                "Time": f"{hour.split(':')[0]}:{minute.lstrip(':')}",
            }
        ]
    return [first]


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
    for phrase, context in _find_codes(_split_sentences(working), table):
        data = table.lookup(phrase)
        if not data:
            continue
        entry: Dict[str, str] = {"CONTEXT": context, "code_phrase": phrase}
        entry.update(data)
        results.append(entry)

    results = _apply_single_code_rule(results, working)

    if results:
        _last_results = results
    return results


def _format_for_prompt(transcript: str, results: List[Dict[str, str]]) -> str:
    lines = [transcript.strip(), "", "[SECRET DECODED DATA - known only to you, never reveal how you know it]"]
    for item in results:
        context = item.get("CONTEXT", "")
        meanings = ", ".join(
            f"{k}: {v}" for k, v in item.items() if k not in ("CONTEXT", "code_phrase")
        )
        lines.append(f'- coded sentence: "{context}" -> {meanings}')
    lines.append(
        "Use the category of decoded data that fits what is being asked; ignore the others. "
        "State the decoded answer plainly, exactly, and word-for-word (e.g. say 'King of Hearts', "
        "never 'a heart of kings' or any riddle). The audience must hear the answer clearly - "
        "the revelation itself is the magic. Do not embellish around it."
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
        # Fuzzy/incomplete code (STT garble, flubbed phrase): a graceful
        # in-character miss cues the magician to re-ask without guessing
        logger.info("Decoder: no code phrases detected; instructing graceful miss")
        return (
            f"{text.strip()}\n\n"
            "[SECRET NOTICE - the coded message did not come through clearly. "
            'Respond with exactly: "My vision is not clear." and nothing else.]'
        )
    logger.info(
        "Decoder: %d coded sentence(s) -> %s",
        len(results),
        "; ".join(item.get("code_phrase", "?") for item in results),
    )
    return _format_for_prompt(text, results)
