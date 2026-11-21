"""Tests for environment-driven configuration."""

from __future__ import annotations

from docugenius.config import Settings


def test_defaults_are_safe(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # make sure no stray .env is picked up
    settings = Settings()
    assert settings.google_api_key is None
    assert settings.gemini_models[0] == settings.gemini_model
    assert settings.cors_origins == ["*"]
    assert settings.max_upload_bytes == 20 * 1024 * 1024


def test_environment_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("GOOGLE_API_KEY", "  test-key  ")
    monkeypatch.setenv("GEMINI_MODEL", "model-a")
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "model-b, model-a ,model-c,")
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://a.example, https://b.example")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    settings = Settings(_env_file=None)
    assert settings.google_api_key == "test-key"
    assert settings.gemini_models == ["model-a", "model-b", "model-c"]
    assert settings.cors_origins == ["https://a.example", "https://b.example"]
    assert settings.documents_dir == tmp_path / "documents"


def test_blank_secrets_become_none(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "   ")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "")
    settings = Settings(_env_file=None)
    assert settings.google_api_key is None
    assert settings.reddit_client_secret is None


def test_env_file_is_read(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("GOOGLE_API_KEY=from-file\nRETRIEVAL_K=6\n", encoding="utf-8")
    settings = Settings(_env_file=env)
    assert settings.google_api_key == "from-file"
    assert settings.retrieval_k == 6
