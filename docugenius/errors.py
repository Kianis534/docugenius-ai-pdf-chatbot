"""Domain errors.

Every error carries an HTTP status code and a message that is safe to show to
end users. The API layer turns them into ``{"detail": "..."}`` responses, so
services never need to import FastAPI.
"""

from __future__ import annotations


class AppError(Exception):
    status_code: int = 500
    default_message: str = "Something went wrong. Please try again."

    def __init__(self, message: str | None = None, *, status_code: int | None = None) -> None:
        self.message = message or self.default_message
        if status_code is not None:
            self.status_code = status_code
        super().__init__(self.message)


# --- LLM ------------------------------------------------------------------- #
class LLMError(AppError):
    status_code = 502
    default_message = "The AI model is temporarily unavailable. Please try again shortly."


class LLMNotConfiguredError(LLMError):
    status_code = 503
    default_message = (
        "Gemini is not configured. Set GOOGLE_API_KEY in your environment (see .env.example) and restart the server."
    )


class LLMQuotaError(LLMError):
    status_code = 429
    default_message = "The Gemini API quota has been reached. Please wait a minute and try again."


class LLMAuthError(LLMError):
    status_code = 502
    default_message = "Gemini rejected the API key. Check that GOOGLE_API_KEY is valid."


# --- Documents ------------------------------------------------------------- #
class DocumentNotFoundError(AppError):
    status_code = 404
    default_message = "Document not found. Please upload it again."


class InvalidDocumentError(AppError):
    status_code = 422
    default_message = "The uploaded file could not be processed."


class PayloadTooLargeError(AppError):
    status_code = 413
    default_message = "The uploaded file is too large."


# --- Audio ----------------------------------------------------------------- #
class TranscriptionError(AppError):
    status_code = 422
    default_message = "The audio could not be transcribed."


# --- Reddit ---------------------------------------------------------------- #
class RedditError(AppError):
    status_code = 502
    default_message = "Reddit could not be reached. Please try again later."


class RedditNotConfiguredError(RedditError):
    status_code = 503
    default_message = (
        "Reddit analysis is not configured. Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET "
        "(see .env.example) and restart the server."
    )
