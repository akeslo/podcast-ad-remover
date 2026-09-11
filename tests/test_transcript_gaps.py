"""
Tests for untranscribed-hole handling.

Whisper (tiny/base, condition_on_previous_text=True) drops the window after a
sign-off line, which is exactly where a dynamically inserted ad sits. Episode
24019 of subscription 1046: transcript ends at 59.92s, next segment starts at
86.16s, the detector returned only the spoken tagline [86.16, 88.64]. These
tests pin the two mitigations: the processor widens ads into adjacent holes,
and the ad prompt renders a marker line so Gemini can see the hole at all.
"""
from unittest.mock import patch

from app.core.ai_services import AdDetector, GAP_MARKER_SECONDS
from app.core.processor import Processor

# Episode 24019 fixture: sign-off ends 59.92, next speech starts 86.16.
TRANSCRIPT_24019 = [
    {"start": 55.0, "end": 59.92, "text": "that's the show, see you tomorrow"},
    {"start": 86.16, "end": 88.64, "text": "book direct at hilton dot com"},
    {"start": 88.64, "end": 90.12, "text": "welcome back"},
]


def absorb(ads, segments, **kw):
    return Processor._absorb_transcript_gaps(ads, segments, **kw)


def spans(result):
    return [[round(s["start"], 2), round(s["end"], 2)] for s in result]


class TestAbsorbTranscriptGaps:
    def test_24019_fixture_extends_start_back_to_hole(self):
        ads = [{"start": 86.16, "end": 88.64, "label": "Ad"}]
        result = absorb(ads, TRANSCRIPT_24019)
        assert spans(result) == [[59.92, 88.64]]
        assert result[0]["label"] == "Ad"

    def test_inputs_are_not_mutated(self):
        ads = [{"start": 86.16, "end": 88.64}]
        absorb(ads, TRANSCRIPT_24019)
        assert ads == [{"start": 86.16, "end": 88.64}]

    def test_hole_under_min_gap_does_not_widen(self):
        segments = [
            {"start": 0.0, "end": 10.0, "text": "a"},
            {"start": 15.0, "end": 20.0, "text": "b"},  # 5s hole
        ]
        ads = [{"start": 15.0, "end": 20.0}]
        assert spans(absorb(segments=segments, ads=ads)) == [[15.0, 20.0]]

    def test_ad_beyond_adj_does_not_widen(self):
        segments = [
            {"start": 0.0, "end": 10.0, "text": "a"},
            {"start": 30.0, "end": 40.0, "text": "b"},  # 20s hole ends at 30
        ]
        ads = [{"start": 33.0, "end": 40.0}]  # starts 3s after hole, adj=2
        assert spans(absorb(ads, segments)) == [[33.0, 40.0]]

    def test_hole_longer_than_max_absorb_is_capped(self):
        segments = [
            {"start": 0.0, "end": 10.0, "text": "a"},
            {"start": 310.0, "end": 320.0, "text": "b"},  # 300s hole
        ]
        ads = [{"start": 310.0, "end": 320.0}]
        assert spans(absorb(ads, segments, max_absorb=120.0)) == [[190.0, 320.0]]

    def test_extends_end_forward_into_following_hole(self):
        segments = [
            {"start": 0.0, "end": 10.0, "text": "a"},
            {"start": 40.0, "end": 50.0, "text": "b"},  # hole 10-40
        ]
        ads = [{"start": 4.0, "end": 9.0}]  # ends 1s before hole
        assert spans(absorb(ads, segments)) == [[4.0, 40.0]]

    def test_never_merges_two_ads_and_keeps_sorted(self):
        segments = [
            {"start": 0.0, "end": 10.0, "text": "a"},
            {"start": 40.0, "end": 50.0, "text": "b"},
        ]
        ads = [{"start": 40.0, "end": 45.0}, {"start": 5.0, "end": 9.0}]
        result = spans(absorb(ads, segments))
        assert len(result) == 2
        assert result == [[5.0, 40.0], [10.0, 45.0]]

    def test_empty_ads(self):
        assert absorb([], TRANSCRIPT_24019) == []


class TestTranscriptRenderingGapMarker:
    def test_marker_emitted_for_long_hole(self):
        assert GAP_MARKER_SECONDS == 8.0
        with patch.object(AdDetector, "_load_settings", return_value={}):
            rendered = AdDetector()._render_transcript(TRANSCRIPT_24019)
        lines = rendered.splitlines()
        assert lines[0] == "[55.00-59.92] that's the show, see you tomorrow"
        assert lines[1] == (
            "[59.92-86.16] <26 s of audio with no transcribed speech "
            "- likely music or a produced ad>"
        )
        assert lines[2] == "[86.16-88.64] book direct at hilton dot com"
        assert len(lines) == 4

    def test_no_marker_for_short_hole(self):
        segments = [
            {"start": 0.0, "end": 10.0, "text": "a"},
            {"start": 13.0, "end": 20.0, "text": "b"},  # 3s hole
        ]
        with patch.object(AdDetector, "_load_settings", return_value={}):
            rendered = AdDetector()._render_transcript(segments)
        assert "no transcribed speech" not in rendered
        assert rendered == "[0.00-10.00] a\n[13.00-20.00] b\n"

    def test_detect_ads_prompt_carries_marker(self):
        with patch.object(AdDetector, "_load_settings", return_value={}):
            detector = AdDetector()
        captured = {}

        class FakeProvider:
            def generate(self, prompt):
                captured["prompt"] = prompt
                return "[]"

        with patch.object(detector, "_get_provider", return_value=FakeProvider()):
            detector.detect_ads({"segments": TRANSCRIPT_24019})
        assert "[59.92-86.16] <26 s of audio with no transcribed speech" in captured["prompt"]
