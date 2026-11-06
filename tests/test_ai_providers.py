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


class TestOllamaProvider:
    def _detector(self, s):
        from app.core.ai_services import AdDetector
        with patch.object(AdDetector, "_load_settings", return_value=s):
            return AdDetector()

    def test_builds_openai_compatible_client_without_key(self):
        with patch("openai.OpenAI") as client:
            prov = self._detector({"ollama_base_url": "http://ks:11434/", "ollama_model": '["gpt-oss:20b"]'}).create_provider("ollama")
        assert prov.models == ["gpt-oss:20b"]
        assert client.call_args.kwargs["base_url"] == "http://ks:11434/v1"
        assert client.call_args.kwargs["timeout"] == 1800

    def test_missing_url_raises(self):
        with patch("app.core.ai_services.settings.OLLAMA_BASE_URL", None):
            with pytest.raises(ValueError, match="Ollama URL"):
                self._detector({}).create_provider("ollama")


class TestHermesProvider:
    def _detector(self, s):
        from app.core.ai_services import AdDetector
        with patch.object(AdDetector, "_load_settings", return_value=s):
            return AdDetector()

    def test_builds_openai_compatible_client_with_key(self):
        with patch("openai.OpenAI") as client:
            prov = self._detector({
                "hermes_base_url": "http://192.168.20.11:8001/",
                "hermes_api_key": "omlx-test-key",
                "hermes_model": '["Qwen3.8-Flash-Next-Uncensored-oQ4e-mtp"]',
            }).create_provider("hermes")
        assert prov.models == ["Qwen3.8-Flash-Next-Uncensored-oQ4e-mtp"]
        assert client.call_args.kwargs["base_url"] == "http://192.168.20.11:8001/v1"
        assert client.call_args.kwargs["timeout"] == 1800

    def test_explicit_key_beats_saved(self):
        with patch("openai.OpenAI") as client:
            self._detector({"hermes_base_url": "http://h:8001", "hermes_api_key": "saved"}).create_provider(
                "hermes", api_key="typed")
        assert client.call_args.kwargs["api_key"] == "typed"

    def test_env_fallback(self):
        with patch("openai.OpenAI") as client:
            with patch("app.core.ai_services.settings.HERMES_BASE_URL", "http://192.168.20.11:8001"), \
                 patch("app.core.ai_services.settings.HERMES_API_KEY", "env-key"):
                self._detector({}).create_provider("hermes")
        assert client.call_args.kwargs["base_url"] == "http://192.168.20.11:8001/v1"
        assert client.call_args.kwargs["api_key"] == "env-key"

    def test_missing_url_raises(self):
        with patch("app.core.ai_services.settings.HERMES_BASE_URL", None):
            with pytest.raises(ValueError, match="oMLX URL"):
                self._detector({}).create_provider("hermes")

    def test_missing_key_raises(self):
        with patch("app.core.ai_services.settings.HERMES_BASE_URL", "http://h:8001"), \
             patch("app.core.ai_services.settings.HERMES_API_KEY", None):
            with pytest.raises(ValueError, match="oMLX API key"):
                self._detector({}).create_provider("hermes")

    def test_long_transcript_is_chunked_like_ollama(self):
        # detect_ads must window long transcripts for hermes too: it is a local
        # server with a fixed context, same silent-truncation hazard as Ollama.
        # 100 segments at 60 s apart (~72k chars) force several windows.
        from app.core.ai_services import AdDetector, OLLAMA_CHUNK_CHARS
        prompts = []

        class P:
            def generate(self, prompt):
                prompts.append(prompt)
                return '[]'

        # keep the _load_settings patch live across detect_ads: it reloads
        # settings at the start of every call
        with patch.object(AdDetector, "_load_settings",
                         return_value={"active_ai_provider": "hermes"}):
            det = AdDetector()
            with patch.object(det, "_get_provider", return_value=P()):
                out = det.detect_ads({"segments": [
                    {"start": float(i * 60), "end": float(i * 60) + 59,
                     "text": "x" * 700} for i in range(100)]})
        assert out == []
        # windowing happened, and every window stayed under the cap
        assert 2 <= len(prompts) <= 4
        assert all(len(p) < OLLAMA_CHUNK_CHARS + 8000 for p in prompts)
