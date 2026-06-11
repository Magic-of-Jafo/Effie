"""Inspect the decoder's structured output for a spoken sentence.

Usage (from corindagpt/):
  ../.venv/Scripts/python -m scripts.decode_inspect "Cool, could you tell me what card this is?"

Prints the decoded category/value data as JSON - exactly what gets embedded
in the secret block for the LLM - without calling any APIs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.services import decoder  # noqa: E402


def main() -> None:
    text = " ".join(sys.argv[1:]) or "Cool, could you tell me what card this is?"
    results = decoder.decode_to_results(text)
    print(json.dumps(results, indent=2))
    print()
    print("--- full prompt block ---")
    print(decoder.decode(text))


if __name__ == "__main__":
    main()
