from __future__ import annotations

import pytest

from src.services import decoder


@pytest.fixture(autouse=True)
def _fresh_decoder_state():
    decoder._last_results = []
    yield
    decoder._last_results = []


def test_simple_code_phrase_number_and_letter():
    results = decoder.decode_to_results("Can you tell me what this is?")
    assert len(results) == 1
    assert results[0]["code_phrase"] == "CAN"
    assert results[0]["Number or date"] == "1"
    assert results[0]["Letter"] == "A"


def test_longest_match_wins_over_substring():
    # "ALL RIGHT NOW CAN" (51) must not decode as "NOW CAN" (11) or "CAN" (1)
    results = decoder.decode_to_results("All right, now can you sense it?")
    assert results[0]["code_phrase"] == "ALL RIGHT NOW CAN"
    assert results[0]["Number or date"] == "51"


def test_playing_card_in_single_phrase():
    results = decoder.decode_to_results("Cool, could you name this card?")
    assert results[0]["code_phrase"] == "COOL COULD"
    assert results[0]["Playing Card Value"] == "King of Hearts"


def test_phrase_spans_stt_punctuation():
    # Real STT output: "Cool. Could you tell me what playing card I'm thinking of?"
    # The period must not break COOL COULD, and the conversational TELL after
    # the match must not register as a second code
    results = decoder.decode_to_results(
        "Cool. Could you tell me what playing card I'm thinking of?"
    )
    assert len(results) == 1
    assert results[0]["code_phrase"] == "COOL COULD"
    assert results[0]["Playing Card Value"] == "King of Hearts"


def test_codes_in_separate_sentences_both_found():
    results = decoder.decode_to_results("Can you feel it? Think hard now.")
    assert [r["code_phrase"] for r in results] == ["CAN", "THINK"]


def test_sorry_resets_previous_codes():
    text = "Well, can you feel it? Sorry. Think hard about this one."
    results = decoder.decode_to_results(text)
    assert len(results) == 1
    assert results[0]["code_phrase"] == "THINK"
    assert results[0]["Number or date"] == "2"


def test_time_combination_across_two_sentences():
    # ARE = hour 4:00, GIVE = minute :05
    results = decoder.decode_to_results("Are you ready to tell us? Give us the full answer.")
    assert results[0].get("Time") == "4:05"
    assert "Hour" not in results[0]


def test_no_match_returns_empty():
    assert decoder.decode_to_results("Nothing coded here at all.") == []


def test_decode_without_codes_instructs_graceful_miss():
    out = decoder.decode("A perfectly innocent sentence.")
    assert out.startswith("A perfectly innocent sentence.")
    assert "My vision is not clear." in out


def test_decode_appends_secret_block():
    out = decoder.decode("Can you hear me?")
    assert out.startswith("Can you hear me?")
    assert "SECRET DECODED DATA" in out
    assert "Number or date: 1" in out


def test_stt_spelling_variant_alright():
    # STT writes "Alright" as one word; the table says "ALL RIGHT"
    results = decoder.decode_to_results("Alright then, tell me what my favorite bird is?")
    assert results[0]["code_phrase"] == "ALL RIGHT THEN TELL"
    assert results[0]["Number or date"] == "79"


def test_punctuation_and_case_insensitivity():
    results = decoder.decode_to_results("NOW... try to remember!")
    assert results[0]["code_phrase"] == "NOW TRY"
    assert results[0]["Number or date"] == "10"
