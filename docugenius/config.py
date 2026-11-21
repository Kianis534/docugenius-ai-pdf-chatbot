"""Application settings, loaded from environment variables and an optional ``.env`` file."""

from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


class Settings(BaseSettings):
    """All runtime configuration. Every field maps to an upper-case environment variable."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Google Gemini ------------------------------------------------------ #
    google_api_key: str | None = Field(default=None, description="Gemini API key from Google AI Studio.")
    gemini_model: str = "gemini-3.8-flash"
    gemini_fallback_models: str = "gemini-3.6-flash,gemini-3.5-flash-lite"
    gemini_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    gemini_timeout_seconds: float = Field(default=60.0, gt=0)
    gemini_daily_request_limit: int = Field(
        default=0, ge=0, description="App-side daily request budget. 0 disables the budget."
    )

    # --- Reddit ------------------------------------------------------------- #
    reddit_client_id: str | None = None
    reddit_client_secret: str | None = None
    reddit_user_agent: str = "python:docugenius:v2.0 (by /u/your_reddit_username)"

    # --- Retrieval-Augmented Generation ------------------------------------ #
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = Field(default=1000, ge=200)
    chunk_overlap: int = Field(default=200, ge=0)
    retrieval_k: int = Field(default=4, ge=1, le=20)
    max_upload_mb: int = Field(default=20, ge=1)

    # --- Server ------------------------------------------------------------- #
    data_dir: Path = Path(tempfile.gettempdir()) / "docugenius"
    allowed_origins: str = "*"
    log_level: str = "INFO"

    @field_validator("google_api_key", "reddit_client_id", "reddit_client_secret", mode="before")
    @classmethod
    def _blank_to_none(cls, value: str | None) -> str | None:
        if isinstance(value, str) and not value.strip():
            return None
        return value.strip() if isinstance(value, str) else value

    @property
    def gemini_models(self) -> list[str]:
        """Primary model followed by fallbacks, de-duplicated and in order."""
        models = [self.gemini_model, *_split_csv(self.gemini_fallback_models)]
        return list(dict.fromkeys(m for m in models if m))

    @property
    def cors_origins(self) -> list[str]:
        return _split_csv(self.allowed_origins) or ["*"]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def documents_dir(self) -> Path:
        return self.data_dir / "documents"


@lru_cache
def get_settings() -> Settings:
    return Settings()
