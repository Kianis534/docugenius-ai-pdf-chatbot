"""Unit tests for the Gemini client: fallback, cooldowns, grounding and budgets."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from google.genai import errors

from docugenius.config import Settings
from docugenius.errors import LLMAuthError, LLMError, LLMNotConfiguredError, LLMQuotaError
from docugenius.services.llm import GeminiClient


def api_error(code: int, message: str, status: str = "ERROR") -> errors.APIError:
    error_class = errors.ClientError if code < 500 else errors.ServerError
    return error_class(code, {"error": {"code": code, "message": message, "status": status}})


def response(text: str = "ok", sources: list[tuple[str, str]] | None = None) -> SimpleNamespace:
    chunks = [SimpleNamespace(web=SimpleNamespace(uri=uri, title=title)) for title, uri in (sources or [])]
    return SimpleNamespace(
        text=text, candidates=[SimpleNamespace(grounding_metadata=SimpleNamespace(grounding_chunks=chunks))]
    )


class FakeModels:
    def __init__(self, outcomes: dict[str, list]) -> None:
        self.outcomes = outcomes
        self.calls: list[tuple[str, object]] = []

    def generate_content(self, *, model, contents, config):
        self.calls.append((model, config))
        queue = self.outcomes[model]
        outcome = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_client(outcomes: dict[str, list], **overrides) -> tuple[GeminiClient, FakeModels]:
    settings = Settings(_env_file=None, gemini_model="primary", gemini_fallback_models="backup,last", **overrides)
    models = FakeModels(outcomes)
    return GeminiClient(settings, client=SimpleNamespace(models=models)), models


def test_not_configured_without_key():
    client = GeminiClient(Settings(_env_file=None))
    assert client.is_configured is False
    with pytest.raises(LLMNotConfiguredError):
        client.generate("hello")


def test_uses_primary_model_when_healthy():
    client, models = make_client({"primary": [response("hi")]})
    result = client.generate("hello", system_instruction="be nice", temperature=0.1)
    assert (result.text, result.model) == ("hi", "primary")
    config = models.calls[0][1]
    assert config.system_instruction == "be nice"
    assert config.temperature == 0.1
    assert config.tools is None


def test_falls_back_on_rate_limit_and_cools_down():
    client, models = make_client(
        {
            "primary": [api_error(429, "Resource has been exhausted", "RESOURCE_EXHAUSTED")],
            "backup": [response("from backup")],
        }
    )
    assert client.generate("hello").model == "backup"
    assert client.status()["cooling_down"] == ["primary"]
    client.generate("again")
    assert [m for m, _ in models.calls] == ["primary", "backup", "backup"]


def test_falls_back_on_retired_model_and_server_errors():
    client, _ = make_client(
        {
            "primary": [api_error(404, "models/primary is not found")],
            "backup": [api_error(503, "The model is overloaded")],
            "last": [response("survivor")],
        }
    )
    assert client.generate("hello").model == "last"


def test_raises_quota_error_when_every_model_is_limited():
    client, _ = make_client({m: [api_error(429, "quota exceeded")] for m in ("primary", "backup", "last")})
    with pytest.raises(LLMQuotaError):
        client.generate("hello")


def test_invalid_api_key_fails_fast():
    client, models = make_client({"primary": [api_error(400, "API key not valid. Please pass a valid API key.")]})
    with pytest.raises(LLMAuthError):
        client.generate("hello")
    assert len(models.calls) == 1


def test_bad_request_is_not_retried_on_other_models():
    client, models = make_client({"primary": [api_error(400, "Request contains an invalid argument.")]})
    with pytest.raises(LLMError):
        client.generate("hello")
    assert len(models.calls) == 1


def test_empty_response_tries_next_model():
    client, _ = make_client({"primary": [response("")], "backup": [response("filled")]})
    assert client.generate("hello").text == "filled"


def test_grounded_search_returns_deduplicated_sources():
    client, models = make_client(
        {
            "primary": [
                response("answer", [("A", "https://a.example"), ("A", "https://a.example"), ("B", "https://b.example")])
            ]
        }
    )
    result = client.generate("news", use_search=True)
    assert result.grounded is True
    assert result.sources == [{"title": "A", "url": "https://a.example"}, {"title": "B", "url": "https://b.example"}]
    assert models.calls[0][1].tools


def test_grounding_rejected_retries_without_search():
    client, models = make_client(
        {"primary": [api_error(400, "Search grounding is not supported for this API key tier."), response("plain")]}
    )
    result = client.generate("news", use_search=True)
    assert (result.text, result.grounded, result.sources) == ("plain", False, [])
    assert models.calls[0][1].tools and models.calls[1][1].tools is None


def test_daily_budget_is_enforced():
    client, _ = make_client({"primary": [response("ok")]}, gemini_daily_request_limit=2)
    client.generate("1")
    client.generate("2")
    with pytest.raises(LLMQuotaError, match="daily"):
        client.generate("3")
    assert client.status()["requests_today"] == 2


def test_network_errors_are_retryable():
    client, _ = make_client({"primary": [TimeoutError("timed out")], "backup": [response("ok")]})
    assert client.generate("hello").model == "backup"
