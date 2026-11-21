"""Google Gemini client with model fallback, cooldowns and an optional daily budget.

Design notes
------------
* The underlying ``google-genai`` client is created lazily, so the app boots
  (and every non-LLM feature works) even when ``GOOGLE_API_KEY`` is missing.
* Models are tried in order. A model that returns 429 (rate limit) is put on a
  short cooldown, and a model that returns 404 (retired or unknown) on a long
  one, so the next request skips straight to a healthy fallback.
* Errors are translated into :mod:`docugenius.errors` types that carry
  user-safe messages and HTTP status codes.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from docugenius.config import Settings
from docugenius.errors import LLMAuthError, LLMError, LLMNotConfiguredError, LLMQuotaError

logger = logging.getLogger(__name__)

RATE_LIMIT_COOLDOWN_SECONDS = 60.0
MISSING_MODEL_COOLDOWN_SECONDS = 6 * 60 * 60.0


@dataclass
class LLMResponse:
    text: str
    model: str
    grounded: bool = False
    sources: list[dict[str, str]] = field(default_factory=list)


class GeminiClient:
    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._settings = settings
        self._client = client
        self._lock = threading.Lock()
        self._cooldown_until: dict[str, float] = {}
        self._budget_day = self._today()
        self._requests_today = 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @property
    def is_configured(self) -> bool:
        return self._client is not None or bool(self._settings.google_api_key)

    @property
    def models(self) -> list[str]:
        return self._settings.gemini_models

    def generate(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        temperature: float | None = None,
        use_search: bool = False,
    ) -> LLMResponse:
        """Generate text, trying each configured model until one succeeds.

        With ``use_search=True`` the request is grounded with Google Search.
        Grounding is a paid-tier feature, so if it is rejected the request is
        retried once without it and the response is marked ``grounded=False``.
        """
        client = self._get_client()
        self._check_budget()

        last_error: LLMError | None = None
        saw_rate_limit = False
        for model in self._available_models():
            try:
                response = self._call(client, model, prompt, system_instruction, temperature, use_search)
                grounded = use_search
            except _GroundingRejected:
                logger.info("Google Search grounding unavailable on %s, retrying without it", model)
                try:
                    response = self._call(client, model, prompt, system_instruction, temperature, False)
                    grounded = False
                except _Retryable as exc:
                    saw_rate_limit |= exc.rate_limited
                    last_error = exc.error
                    continue
            except _Retryable as exc:
                saw_rate_limit |= exc.rate_limited
                last_error = exc.error
                continue

            text = (getattr(response, "text", None) or "").strip()
            if not text:
                logger.warning("Gemini model %s returned an empty response", model)
                last_error = LLMError("The AI model returned an empty response. Try rephrasing your request.")
                continue

            self._record_success()
            sources = self._extract_sources(response) if grounded else []
            return LLMResponse(text=text, model=model, grounded=grounded and bool(sources), sources=sources)

        if saw_rate_limit:
            raise LLMQuotaError()
        raise last_error or LLMError()

    def status(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            self._roll_budget_day()
            cooling = sorted(m for m, until in self._cooldown_until.items() if until > now)
            return {
                "configured": self.is_configured,
                "models": self.models,
                "cooling_down": cooling,
                "requests_today": self._requests_today,
                "daily_limit": self._settings.gemini_daily_request_limit or None,
            }

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self._settings.google_api_key:
            raise LLMNotConfiguredError()
        from google import genai
        from google.genai import types

        self._client = genai.Client(
            api_key=self._settings.google_api_key,
            http_options=types.HttpOptions(timeout=int(self._settings.gemini_timeout_seconds * 1000)),
        )
        return self._client

    def _call(
        self,
        client: Any,
        model: str,
        prompt: str,
        system_instruction: str | None,
        temperature: float | None,
        use_search: bool,
    ) -> Any:
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=self._settings.gemini_temperature if temperature is None else temperature,
            tools=[types.Tool(google_search=types.GoogleSearch())] if use_search else None,
        )
        try:
            return client.models.generate_content(model=model, contents=prompt, config=config)
        except errors.APIError as exc:
            raise self._classify(exc, model, use_search) from exc
        except Exception as exc:  # network errors, timeouts, etc.
            logger.warning("Gemini request to %s failed: %s", model, exc)
            raise _Retryable(LLMError()) from exc

    def _classify(self, exc: Any, model: str, use_search: bool) -> Exception:
        code = getattr(exc, "code", None) or 0
        message = (getattr(exc, "message", None) or str(exc)).lower()
        logger.warning("Gemini %s returned %s: %s", model, code, message[:300])

        if code == 429 or "resource_exhausted" in message or "quota" in message:
            if use_search and ("search" in message or "grounding" in message):
                return _GroundingRejected()
            self._cool_down(model, RATE_LIMIT_COOLDOWN_SECONDS)
            return _Retryable(LLMQuotaError(), rate_limited=True)
        if code == 404 or "not found" in message or "no longer available" in message:
            self._cool_down(model, MISSING_MODEL_COOLDOWN_SECONDS)
            return _Retryable(LLMError(f"The Gemini model '{model}' is not available."))
        if use_search and code in (400, 403) and ("search" in message or "grounding" in message or "tool" in message):
            return _GroundingRejected()
        if code == 401 or "api key" in message or "api_key" in message:
            return LLMAuthError()
        if code == 403:
            return _Retryable(
                LLMAuthError(
                    "Gemini denied access (HTTP 403). Check that GOOGLE_API_KEY is valid, not revoked, "
                    "and allowed to use the configured models."
                )
            )
        if code >= 500 or code in (408, 499):
            return _Retryable(LLMError())
        return LLMError("Gemini could not process this request. Try rephrasing it or using shorter input.")

    def _available_models(self) -> list[str]:
        now = time.monotonic()
        with self._lock:
            ready = [m for m in self.models if self._cooldown_until.get(m, 0) <= now]
        # If every model is cooling down, try them all anyway rather than failing outright.
        return ready or list(self.models)

    def _cool_down(self, model: str, seconds: float) -> None:
        with self._lock:
            self._cooldown_until[model] = time.monotonic() + seconds

    @staticmethod
    def _today() -> str:
        return datetime.now(UTC).date().isoformat()

    def _roll_budget_day(self) -> None:
        today = self._today()
        if today != self._budget_day:
            self._budget_day = today
            self._requests_today = 0

    def _check_budget(self) -> None:
        limit = self._settings.gemini_daily_request_limit
        with self._lock:
            self._roll_budget_day()
            if limit and self._requests_today >= limit:
                raise LLMQuotaError("This server's daily AI request budget has been used up. It resets at 00:00 UTC.")

    def _record_success(self) -> None:
        with self._lock:
            self._roll_budget_day()
            self._requests_today += 1

    @staticmethod
    def _extract_sources(response: Any) -> list[dict[str, str]]:
        sources: list[dict[str, str]] = []
        seen: set[str] = set()
        for candidate in getattr(response, "candidates", None) or []:
            metadata = getattr(candidate, "grounding_metadata", None)
            for chunk in getattr(metadata, "grounding_chunks", None) or []:
                web = getattr(chunk, "web", None)
                uri = getattr(web, "uri", None)
                if uri and uri not in seen:
                    seen.add(uri)
                    sources.append({"title": getattr(web, "title", None) or uri, "url": uri})
        return sources


class _Retryable(Exception):
    def __init__(self, error: LLMError, rate_limited: bool = False) -> None:
        super().__init__(error.message)
        self.error = error
        self.rate_limited = rate_limited


class _GroundingRejected(Exception):
    pass
