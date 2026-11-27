"""The AI summary is read aloud and must stay short: prompt rule plus a hard word cap."""
from unittest.mock import MagicMock, patch

from app.core.ai_services import (
    AdDetector, MAX_SUMMARY_WORDS, SUMMARY_LENGTH_RULE, limit_summary_words,
)

# Measured Piper en_US-amy-medium speed is ~170 wpm; 20 s allows ~56 words.
SLOWEST_PLAUSIBLE_WPM = 140


def test_cap_keeps_spoken_length_under_twenty_seconds():
    assert MAX_SUMMARY_WORDS / SLOWEST_PLAUSIBLE_WPM * 60 <= 20


def test_short_text_is_untouched():
    assert limit_summary_words("This episode includes cats.") == "This episode includes cats."


def test_long_text_is_cut_to_cap_at_a_sentence_end():
    text = " ".join(["Sentence number %d has six words." % i for i in range(40)])
    out = limit_summary_words(text)
    assert len(out.split()) <= MAX_SUMMARY_WORDS
    assert out.endswith(".")
    assert out.split(".")[-1] == ""


def test_run_on_text_is_cut_hard():
    out = limit_summary_words("word " * 500)
    assert len(out.split()) <= MAX_SUMMARY_WORDS
    assert out.endswith(".")


def test_generate_summary_enforces_cap_and_adds_length_rule():
    det = AdDetector.__new__(AdDetector)
    det._load_settings = lambda: {"summary_prompt_template": "Summarize: {transcript_context}"}
    provider = MagicMock()
    provider.generate.return_value = "blah " * 400
    det._get_provider = lambda: provider
    out = det.generate_summary({"segments": [{"text": "hello"}]}, "p", "t", "d")
    assert len(out.split()) <= MAX_SUMMARY_WORDS
    assert SUMMARY_LENGTH_RULE in provider.generate.call_args[0][0]
