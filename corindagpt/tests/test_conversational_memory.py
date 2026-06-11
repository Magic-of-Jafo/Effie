from __future__ import annotations

from src.services.conversational_memory import ConversationalMemory, build_memory


def test_exchange_recorded_in_order():
    m = ConversationalMemory(max_turns=5)
    m.add_exchange("What card am I holding?", "King of Hearts.")
    msgs = m.messages()
    assert msgs == [
        {"role": "user", "content": "What card am I holding?"},
        {"role": "assistant", "content": "King of Hearts."},
    ]


def test_played_phrase_counts_as_effie_speaking():
    m = ConversationalMemory()
    m.add_assistant("I dreamt of the stars again.")
    assert m.messages() == [{"role": "assistant", "content": "I dreamt of the stars again."}]


def test_rolling_window_drops_oldest():
    m = ConversationalMemory(max_turns=2)  # keeps 4 messages
    for i in range(5):
        m.add_exchange(f"q{i}", f"a{i}")
    msgs = m.messages()
    assert len(msgs) == 4
    assert msgs[0]["content"] == "q3"
    assert msgs[-1]["content"] == "a4"


def test_blank_entries_ignored():
    m = ConversationalMemory()
    m.add_user("")
    m.add_assistant("   ")
    assert m.messages() == []


def test_build_memory_reads_config():
    m = build_memory({"memory": {"max_turns": 3}})
    assert m.max_turns == 3


def test_clear():
    m = ConversationalMemory()
    m.add_exchange("q", "a")
    m.clear()
    assert m.messages() == []
