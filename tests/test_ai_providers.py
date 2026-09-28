"""
Tests for OpenAIProvider/AnthropicProvider's multi-key rotation and transient-error retry
(enhancements.md E-20260915-1) — the resilience GeminiProvider already had, extended to the
"currently have simpler flat model-list fallback with no multi-key rotation or
transient-error retry" providers named in README's Features section.

Real SDK clients (openai.OpenAI / anthropic.Anthropic) are always mocked; no network call.
"""
from unittest.mock import MagicMock, patch

import pytest

from app.core.ai_services import AnthropicProvider, OpenAIProvider, RateLimitError, TRANSIENT_ERROR_PATTERNS, RATE_LIMIT_ERROR_PATTERNS, _matches_any


def _chat_response(text):
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=text))]
    return resp


def _anthropic_response(text):
    resp = MagicMock()
    resp.content = [MagicMock(text=text)]
    return resp


class TestPatternClassification:
    def test_transient_patterns_match_common_5xx_and_timeout_strings(self):
        assert _matches_any("503 Service Unavailable", TRANSIENT_ERROR_PATTERNS)
        assert _matches_any("Request timed out", TRANSIENT_ERROR_PATTERNS)
        assert not _matches_any("invalid api key", TRANSIENT_ERROR_PATTERNS)

    def test_rate_limit_patterns_match_429_and_quota_strings(self):
        assert _matches_any("429 Too Many Requests", RATE_LIMIT_ERROR_PATTERNS)
        assert _matches_any("Rate limit exceeded", RATE_LIMIT_ERROR_PATTERNS)
        assert not _matches_any("model not found", RATE_LIMIT_ERROR_PATTERNS)


class TestOpenAIProviderKeyRotation:
    @patch("openai.OpenAI")
    def test_accepts_a_bare_string_key_like_before_rotation_existed(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _chat_response("hi")

        provider = OpenAIProvider("sk-single-key", ["gpt-4o"])
        assert provider.generate("hello") == "hi"
        assert provider.api_keys == ["sk-single-key"]

    @patch("openai.OpenAI")
    def test_rotates_to_next_key_when_every_model_is_rate_limited_on_the_first(self, mock_openai_cls):
        client_a = MagicMock()
        client_b = MagicMock()
        mock_openai_cls.side_effect = [client_a, client_b]
        client_a.chat.completions.create.side_effect = Exception("429 Too Many Requests")
        client_b.chat.completions.create.return_value = _chat_response("ok on key 2")

        provider = OpenAIProvider(["key-a", "key-b"], ["gpt-4o"])
        result = provider.generate("hello")

        assert result == "ok on key 2"
        assert provider.current_key_idx == 1

    @patch("openai.OpenAI")
    def test_raises_ratelimiterror_when_all_keys_and_models_are_rate_limited(self, mock_openai_cls):
        client = MagicMock()
        mock_openai_cls.return_value = client
        client.chat.completions.create.side_effect = Exception("429 rate limit")

        provider = OpenAIProvider(["key-a", "key-b"], ["gpt-4o", "gpt-4o-mini"])
        with pytest.raises(RateLimitError):
            provider.generate("hello")

    @patch("time.sleep", return_value=None)
    @patch("openai.OpenAI")
    def test_retries_the_same_model_in_place_on_a_transient_error_then_succeeds(self, mock_openai_cls, _sleep):
        client = MagicMock()
        mock_openai_cls.return_value = client
        client.chat.completions.create.side_effect = [
            Exception("503 Service Unavailable"),
            _chat_response("recovered"),
        ]

        provider = OpenAIProvider("key-a", ["gpt-4o"])
        assert provider.generate("hello") == "recovered"
        assert client.chat.completions.create.call_count == 2

    @patch("openai.OpenAI")
    def test_a_non_transient_non_rate_limit_error_falls_through_to_the_next_model_without_rotating_keys(self, mock_openai_cls):
        client = MagicMock()
        mock_openai_cls.return_value = client
        client.chat.completions.create.side_effect = [
            Exception("400 bad request: invalid model"),
            _chat_response("second model worked"),
        ]

        provider = OpenAIProvider("key-a", ["bad-model", "gpt-4o"])
        assert provider.generate("hello") == "second model worked"
        assert provider.current_key_idx == 0  # never rotated — this wasn't a rate limit


class TestAnthropicProviderKeyRotation:
    @patch("anthropic.Anthropic")
    def test_accepts_a_bare_string_key_like_before_rotation_existed(self, mock_anthropic_cls):
        client = MagicMock()
        mock_anthropic_cls.return_value = client
        client.messages.create.return_value = _anthropic_response("hi")

        provider = AnthropicProvider("sk-ant-single", ["claude-3-5-sonnet-20241022"])
        assert provider.generate("hello") == "hi"
        assert provider.api_keys == ["sk-ant-single"]

    @patch("anthropic.Anthropic")
    def test_rotates_to_next_key_when_rate_limited_on_the_first(self, mock_anthropic_cls):
        client_a = MagicMock()
        client_b = MagicMock()
        mock_anthropic_cls.side_effect = [client_a, client_b]
        client_a.messages.create.side_effect = Exception("rate_limit_error: 429")
        client_b.messages.create.return_value = _anthropic_response("ok on key 2")

        provider = AnthropicProvider(["key-a", "key-b"], ["claude-3-5-sonnet-20241022"])
        result = provider.generate("hello")

        assert result == "ok on key 2"
        assert provider.current_key_idx == 1

    @patch("anthropic.Anthropic")
    def test_raises_ratelimiterror_when_all_keys_exhausted(self, mock_anthropic_cls):
        client = MagicMock()
        mock_anthropic_cls.return_value = client
        client.messages.create.side_effect = Exception("429 too many requests")

        provider = AnthropicProvider(["key-a"], ["claude-3-5-sonnet-20241022"])
        with pytest.raises(RateLimitError):
            provider.generate("hello")

    @patch("time.sleep", return_value=None)
    @patch("anthropic.Anthropic")
    def test_retries_the_same_model_in_place_on_a_transient_error_then_succeeds(self, mock_anthropic_cls, _sleep):
        client = MagicMock()
        mock_anthropic_cls.return_value = client
        client.messages.create.side_effect = [
            Exception("500 internal server error"),
            _anthropic_response("recovered"),
        ]

        provider = AnthropicProvider("key-a", ["claude-3-5-sonnet-20241022"])
        assert provider.generate("hello") == "recovered"
        assert client.messages.create.call_count == 2


class TestAnthropicProviderListModels:
    """
    list_models() used to always return a hardcoded, unmaintained list of 2024-era model ids
    (the admin "refresh models" dropdown at /admin/ai/refresh/anthropic never showed a model
    released after that point). It now calls the live API, matching OpenAIProvider's pattern,
    falling back to the old static list only when the call itself fails.
    """

    @patch("anthropic.Anthropic")
    def test_lists_live_models_from_the_api_instead_of_the_hardcoded_2024_list(self, mock_anthropic_cls):
        client = MagicMock()
        mock_anthropic_cls.return_value = client
        client.models.list.return_value = MagicMock(data=[
            MagicMock(id="claude-sonnet-4-5-20260929"),
            MagicMock(id="claude-3-5-sonnet-20241022"),
        ])

        provider = AnthropicProvider("key-a", ["claude-3-5-sonnet-20241022"])
        models = provider.list_models()

        assert "claude-sonnet-4-5-20260929" in models
        client.models.list.assert_called_once()

    @patch("anthropic.Anthropic")
    def test_falls_back_to_the_static_list_when_the_live_call_fails(self, mock_anthropic_cls):
        client = MagicMock()
        mock_anthropic_cls.return_value = client
        client.models.list.side_effect = Exception("network error")

        provider = AnthropicProvider("key-a", ["claude-3-5-sonnet-20241022"])
        models = provider.list_models()

        assert models == AnthropicProvider._FALLBACK_MODELS
        assert models is not AnthropicProvider._FALLBACK_MODELS  # caller gets a copy, not the shared list
