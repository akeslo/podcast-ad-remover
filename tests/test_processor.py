"""
Tests for app/core/processor.py - the core ad-removal pipeline (9 churn
touches per PROP-077, previously zero coverage).

Transcriber and AdDetector (both of which wrap real external AI services) and
the repositories (both of which wrap the real SQLite database) are always
replaced with mocks before `_process_episode_inner`/`process_episode` are
exercised, so these tests never make a real AI call and never touch the real
DB.
"""
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.ai_services import RateLimitError
from app.core.config import settings
from app.core.models import Episode
from app.core.processor import Processor

FAKE_TRANSCRIPT = {
    "text": "hello world this is an ad break the end",
    "segments": [
        {"start": 0.0, "end": 5.0, "text": "hello world"},
        {"start": 5.0, "end": 10.0, "text": "this is an ad break"},
        {"start": 10.0, "end": 15.0, "text": "the end"},
    ],
    "language": "en",
}


def make_episode(**overrides) -> Episode:
    defaults = dict(
        id=1,
        subscription_id=1,
        guid="episode-guid-1",
        title="Test Episode",
        pub_date=None,
        original_url="https://example.com/episode.mp3",
        duration=900,
        status="processing",
        processing_flags=None,
        transcript_path=None,
    )
    defaults.update(overrides)
    return Episode(**defaults)


def make_subscription(**overrides) -> SimpleNamespace:
    defaults = dict(
        id=1,
        slug="test-podcast",
        title="Test Podcast",
        remove_ads=True,
        remove_promos=True,
        remove_intros=False,
        remove_outros=False,
        custom_instructions=None,
        append_title_intro=False,
        ai_rewrite_description=False,
        append_summary=False,
        ai_audio_summary=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def make_mocked_processor() -> Processor:
    """A Processor whose AI clients and repositories are all mocks."""
    processor = Processor()
    processor.ep_repo = MagicMock()
    processor.ep_repo.get_status.return_value = "processing"
    processor.sub_repo = MagicMock()
    processor.transcriber = MagicMock()
    processor.ad_detector = MagicMock()
    processor.ad_detector.generate_audio = AsyncMock()
    processor.ad_detector.validate_tts = AsyncMock()
    processor.rss_gen = MagicMock()
    return processor


def seed_pre_downloaded_audio(sub_slug: str, guid: str) -> str:
    """Pre-create the input audio file so _process_episode_inner's download
    branch (a real httpx network call) is never reached."""
    episode_slug = guid.replace("/", "_").replace(" ", "_")
    episode_dir = settings.get_episode_dir(sub_slug, episode_slug)
    os.makedirs(episode_dir, exist_ok=True)
    input_path = os.path.join(episode_dir, "original.mp3")
    with open(input_path, "wb") as f:
        f.write(b"fake-mp3-bytes")
    return episode_dir


class TestExtractText:
    def test_extracts_overlapping_segments(self):
        processor = Processor()
        segments = [
            {"start": 0.0, "end": 5.0, "text": "hello"},
            {"start": 5.0, "end": 10.0, "text": "world"},
            {"start": 10.0, "end": 15.0, "text": "unrelated"},
        ]
        result = processor._extract_text(2.0, 8.0, segments)
        assert result == "hello world"

    def test_no_overlap_returns_empty_string(self):
        processor = Processor()
        segments = [{"start": 0.0, "end": 5.0, "text": "hello"}]
        assert processor._extract_text(10.0, 20.0, segments) == ""


class TestCheckCancellation:
    def test_returns_true_when_still_processing(self):
        processor = make_mocked_processor()
        ep = make_episode()
        assert processor._check_cancellation(ep) is True

    def test_returns_false_and_cleans_up_when_status_changed(self):
        processor = make_mocked_processor()
        processor.ep_repo.get_status.return_value = "unprocessed"
        processor.sub_repo.get_by_id.return_value = make_subscription()
        ep = make_episode()

        assert processor._check_cancellation(ep) is False
        processor.sub_repo.get_by_id.assert_called_once_with(ep.subscription_id)


class TestProcessEpisode:
    @pytest.mark.asyncio
    async def test_resets_status_to_pending_and_triggers_queue(self):
        processor = make_mocked_processor()
        processor.process_queue = AsyncMock()

        await processor.process_episode(42)

        processor.ep_repo.update_status.assert_called_once_with(42, "pending")
        processor.process_queue.assert_awaited_once()


class TestProcessEpisodeInner:
    @pytest.mark.asyncio
    async def test_happy_path_marks_episode_completed(self):
        processor = make_mocked_processor()
        sub = make_subscription()
        ep = make_episode()
        seed_pre_downloaded_audio(sub.slug, ep.guid)

        processor.transcriber.transcribe = MagicMock(return_value=FAKE_TRANSCRIPT)
        processor.ad_detector.detect_ads = MagicMock(
            return_value=[
                {"start": 5.0, "end": 10.0, "label": "Ad", "reason": "sponsor read"}
            ]
        )

        with patch("app.core.processor.AudioProcessor.remove_segments") as remove_segments:
            await processor._process_episode_inner(ep, sub, {"retry_count": 0})

        remove_segments.assert_called_once()
        processor.ad_detector.detect_ads.assert_called_once()
        processor.transcriber.transcribe.assert_called_once()

        # Final status transition to "completed", not a rate-limit/retry path.
        completed_calls = [
            call
            for call in processor.ep_repo.update_status.call_args_list
            if call.args[:2] == (ep.id, "completed")
        ]
        assert len(completed_calls) == 1

        processor.rss_gen.generate_feed.assert_called_once_with(sub.id)
        processor.rss_gen.generate_unified_feed.assert_called_once()

        # Text summary / audio summary / title intro are all disabled on this
        # subscription, so the AI summary/TTS paths must never be touched.
        processor.ad_detector.generate_audio.assert_not_awaited()
        processor.ad_detector.validate_tts.assert_not_awaited()
        processor.ep_repo.update_ai_summary.assert_not_called()

    @pytest.mark.asyncio
    async def test_rate_limit_error_places_episode_on_hold(self):
        processor = make_mocked_processor()
        sub = make_subscription()
        ep = make_episode()
        seed_pre_downloaded_audio(sub.slug, ep.guid)

        processor.transcriber.transcribe = MagicMock(return_value=FAKE_TRANSCRIPT)
        processor.ad_detector.detect_ads = MagicMock(
            side_effect=RateLimitError("quota exceeded", is_daily_limit=True, provider="gemini")
        )

        with patch("app.core.processor.AudioProcessor.remove_segments"):
            await processor._process_episode_inner(ep, sub, {"retry_count": 0})

        processor.ep_repo.update_rate_limited.assert_called_once()
        args = processor.ep_repo.update_rate_limited.call_args.args
        assert args[0] == ep.id
        # Never fell through to the generic failure/retry path, and never
        # reached the "completed"/"failed" terminal status transitions
        # (only the initial "processing" mark from the top of the method).
        processor.ep_repo.update_retry.assert_not_called()
        statuses = [call.args[1] for call in processor.ep_repo.update_status.call_args_list]
        assert statuses == ["processing"]

    @pytest.mark.asyncio
    async def test_generic_error_schedules_retry(self):
        processor = make_mocked_processor()
        sub = make_subscription()
        ep = make_episode()
        seed_pre_downloaded_audio(sub.slug, ep.guid)

        processor.transcriber.transcribe = MagicMock(side_effect=RuntimeError("boom"))

        await processor._process_episode_inner(ep, sub, {"retry_count": 0})

        processor.ep_repo.update_retry.assert_called_once()
        retry_call_args = processor.ep_repo.update_retry.call_args.args
        assert retry_call_args[0] == ep.id
        assert retry_call_args[1] == 1  # retry_count incremented from 0 -> 1
        # Only the initial "processing" mark - never reached "completed"/"failed".
        statuses = [call.args[1] for call in processor.ep_repo.update_status.call_args_list]
        assert statuses == ["processing"]
        processor.ep_repo.update_rate_limited.assert_not_called()

    @pytest.mark.asyncio
    async def test_generic_error_matching_rate_limit_pattern_is_treated_as_rate_limit(self):
        processor = make_mocked_processor()
        sub = make_subscription()
        ep = make_episode()
        seed_pre_downloaded_audio(sub.slug, ep.guid)

        processor.transcriber.transcribe = MagicMock(
            side_effect=RuntimeError("429 Too Many Requests")
        )

        await processor._process_episode_inner(ep, sub, {"retry_count": 0})

        processor.ep_repo.update_rate_limited.assert_called_once()
        processor.ep_repo.update_retry.assert_not_called()

    @pytest.mark.asyncio
    async def test_max_retries_exceeded_marks_episode_failed(self):
        processor = make_mocked_processor()
        sub = make_subscription()
        ep = make_episode()
        seed_pre_downloaded_audio(sub.slug, ep.guid)

        processor.transcriber.transcribe = MagicMock(side_effect=RuntimeError("boom"))

        await processor._process_episode_inner(ep, sub, {"retry_count": 5})

        # First call marks "processing" (top of the method), second call is
        # the terminal "failed" transition once retries are exhausted.
        statuses = [call.args[1] for call in processor.ep_repo.update_status.call_args_list]
        assert statuses == ["processing", "failed"]
        processor.ep_repo.update_status.assert_any_call(ep.id, "failed", error="boom")
        processor.ep_repo.update_retry.assert_not_called()
