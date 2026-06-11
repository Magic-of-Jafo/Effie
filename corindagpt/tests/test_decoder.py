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
    results = decoder.decode_to_results("Cool. Could you name this card?")
    # Sentence split puts "Cool." alone; "Could..." alone -> COULD is not a
    # phrase by itself, so test the unsplit form too
    results = decoder.decode_to_results("Cool, could you name this card?")
    assert results[0]["code_phrase"] == "COOL COULD"
    assert results[0]["Playing Card Value"] == "King of Hearts"


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


def test_no_match_returns_empty_then_remembers_last():
    assert decoder.decode_to_results("Nothing coded here at all.") == []
    decoder.decode_to_results("Can you hear me?")
    repeated = decoder.decode_to_results("Nothing coded here either.")
    assert repeated and repeated[0]["code_phrase"] == "CAN"


def test_decode_passthrough_without_codes():
    text = "A perfectly innocent sentence."
    assert decoder.decode(text) == text


def test_decode_appends_secret_block():
    out = decoder.decode("Can you hear me?")
    assert out.startswith("Can you hear me?")
    assert "SECRET DECODED DATA" in out
    assert "Number or date: 1" in out


def test_punctuation_and_case_insensitivity():
    results = decoder.decode_to_results("NOW... try to remember!")
    assert results[0]["code_phrase"] == "NOW TRY"
    assert results[0]["Number or date"] == "10"
