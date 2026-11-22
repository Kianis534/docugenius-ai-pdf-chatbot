"""Tests for system, assistant, audio, Reddit, ghostwriter and sketch endpoints."""

from __future__ import annotations

from conftest import make_wav

from docugenius import __version__
from docugenius.errors import LLMError


# --- System ---------------------------------------------------------------- #
def test_health_reports_services(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__
    assert body["services"]["gemini"]["configured"] is True
    assert body["services"]["reddit"]["configured"] is True
    assert body["services"]["documents"]["indexed"] == 0
    assert "ffmpeg" in body["services"]["speech"]


def test_frontend_and_assets_are_served(client):
    index = client.get("/")
    assert index.status_code == 200
    assert "DocuGenius" in index.text
    assert client.get("/assets/js/app.js").status_code == 200
    assert client.get("/assets/css/styles.css").status_code == 200
    assert client.get("/assets/animations/ai_animation.json").status_code == 200


def test_source_files_are_not_exposed(client):
    for path in ("/assets/../docugenius/config.py", "/static/docugenius/main.py", "/.env", "/assets/.env"):
        assert client.get(path).status_code == 404, path


def test_openapi_schema_lists_all_routes(client):
    paths = client.get("/openapi.json").json()["paths"]
    for route in (
        "/health",
        "/api/documents",
        "/api/chat",
        "/api/translate",
        "/api/search",
        "/api/transcribe",
        "/api/reddit/sentiment",
        "/api/ghostwriter",
        "/api/sketch/enhance",
    ):
        assert route in paths


# --- Chat / translate / search -------------------------------------------- #
def test_general_chat_includes_history(client, fake_llm):
    response = client.post(
        "/api/chat",
        json={
            "message": "And in French?",
            "history": [{"role": "user", "content": "Say hello"}, {"role": "assistant", "content": "Hello!"}],
        },
    )
    assert response.status_code == 200
    assert response.json()["mode"] == "general"
    prompt = fake_llm.calls[-1]["prompt"]
    assert "User: Say hello" in prompt and "Assistant: Hello!" in prompt and "User: And in French?" in prompt


def test_document_chat_uses_rag(client, uploaded):
    response = client.post("/api/chat", json={"message": "What is the outlook?", "doc_id": uploaded["doc_id"]})
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "document"
    assert body["sources"]


def test_chat_rejects_bad_input(client):
    assert client.post("/api/chat", json={"message": ""}).status_code == 422
    assert client.post("/api/chat", json={"message": "hi", "doc_id": "../../etc"}).status_code == 422


def test_translate(client, fake_llm):
    fake_llm.reply = "علم طاقت ہے"
    response = client.post("/api/translate", json={"text": "Knowledge is power", "target_language": "Urdu"})
    assert response.status_code == 200
    assert response.json() == {"translation": "علم طاقت ہے", "target_language": "Urdu", "model": "fake-model"}
    assert "into Urdu" in fake_llm.calls[-1]["prompt"]


def test_search_requests_grounding_and_returns_sources(client, fake_llm):
    fake_llm.grounded = True
    fake_llm.sources = [{"title": "Example", "url": "https://example.com"}]
    response = client.post("/api/search", json={"query": "latest RAG research"})
    assert response.status_code == 200
    assert fake_llm.calls[-1]["use_search"] is True
    assert response.json()["grounded"] is True
    assert response.json()["sources"] == [{"title": "Example", "url": "https://example.com"}]


# --- Audio ----------------------------------------------------------------- #
def test_transcribe_wav(client, fake_recognizer):
    response = client.post(
        "/api/transcribe",
        files={"file": ("clip.wav", make_wav(), "audio/wav")},
        data={"language": "ur-PK"},
    )
    assert response.status_code == 200, response.text
    assert response.json() == {"text": "hello from docugenius", "language": "ur-PK"}
    assert fake_recognizer.languages == ["ur-PK"]


def test_transcribe_non_wav_without_ffmpeg(client, monkeypatch):
    from docugenius.services.speech import Transcriber

    monkeypatch.setattr(Transcriber, "ffmpeg_available", staticmethod(lambda: False))
    response = client.post("/api/transcribe", files={"file": ("clip.mp3", b"ID3fakeaudio", "audio/mpeg")})
    assert response.status_code == 415
    assert "FFmpeg" in response.json()["detail"]


def test_transcribe_rejects_invalid_language(client):
    response = client.post(
        "/api/transcribe", files={"file": ("clip.wav", make_wav(), "audio/wav")}, data={"language": "<script>"}
    )
    assert response.status_code == 422


# --- Reddit ---------------------------------------------------------------- #
def test_reddit_sentiment(client, fake_reddit, fake_llm):
    fake_llm.reply = "- Mostly upbeat about EVs"
    response = client.post("/api/reddit/sentiment", json={"subject": "electric cars", "time_filter": "month"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total_posts"] == 5  # 3 submissions + 2 comments
    assert round(body["positive"] + body["neutral"] + body["negative"]) == 100
    assert body["chart"].startswith("data:image/png;base64,")
    assert body["analysis"] == "- Mostly upbeat about EVs"
    assert body["analysis_engine"] == "gemini"
    assert body["top_subreddits"] == {"cars": 2, "technology": 3}
    assert fake_reddit.queries[-1] == {"query": "electric cars", "time_filter": "month", "limit": 25}


def test_reddit_falls_back_to_heuristic_summary(client, fake_llm):
    fake_llm.error = LLMError()
    body = client.post("/api/reddit/sentiment", json={"subject": "electric cars"}).json()
    assert body["analysis_engine"] == "heuristic"
    assert "Overall mood" in body["analysis"]


def test_reddit_not_configured(client, services, settings):
    from docugenius.services.reddit import RedditSentimentAnalyzer

    services.reddit = RedditSentimentAnalyzer(settings, services.llm)
    response = client.post("/api/reddit/sentiment", json={"subject": "electric cars"})
    assert response.status_code == 503
    assert "REDDIT_CLIENT_ID" in response.json()["detail"]


def test_reddit_no_results(client, fake_reddit):
    fake_reddit.submissions = []
    assert client.post("/api/reddit/sentiment", json={"subject": "zzzz"}).status_code == 404


# --- Ghostwriter ----------------------------------------------------------- #
def test_ghostwriter_uses_gemini_when_available(client, fake_llm):
    fake_llm.reply = "We must resolve this issue promptly."
    response = client.post("/api/ghostwriter", json={"text": "We need to fix this problem quickly.", "style": "formal"})
    assert response.status_code == 200
    assert response.json()["engine"] == "gemini"
    assert response.json()["rewritten"] == "We must resolve this issue promptly."


def test_ghostwriter_falls_back_to_rules(client, fake_llm):
    fake_llm.configured = False
    response = client.post("/api/ghostwriter", json={"text": "We need to fix this problem quickly.", "style": "formal"})
    body = response.json()
    assert body["engine"] == "rules"
    assert body["rewritten"] == "We must rectify this issue promptly."


def test_ghostwriter_validation(client):
    assert client.post("/api/ghostwriter", json={"text": "hi", "style": "pirate"}).status_code == 422
    assert client.post("/api/ghostwriter", json={"text": "   ", "style": "formal"}).status_code == 422


# --- Sketch ---------------------------------------------------------------- #
def test_sketch_enhance(client):
    import math

    circle = [{"x": 100 + 50 * math.cos(t / 10), "y": 100 + 50 * math.sin(t / 10)} for t in range(64)]
    line = [{"x": float(x), "y": 10.0 + (x % 2)} for x in range(0, 200, 5)]
    response = client.post("/api/sketch/enhance", json={"strokes": [{"points": circle}, {"points": line}]})
    assert response.status_code == 200
    body = response.json()
    assert [s["type"] for s in body["strokes"]] == ["circle", "line"]
    assert body["summary"] == {"circle": 1, "line": 1}
    assert body["message"] == "Cleaned up 1 circle, 1 line."


def test_sketch_requires_strokes(client):
    assert client.post("/api/sketch/enhance", json={"strokes": []}).status_code == 422


def test_health_exposes_upload_limits(client):
    limits = client.get("/health").json()["limits"]
    assert limits == {"max_upload_mb": 2, "max_audio_mb": 25}


def test_security_headers_are_set(client):
    for path in ("/", "/health", "/assets/js/app.js"):
        headers = client.get(path).headers
        assert headers["x-content-type-options"] == "nosniff", path
        assert headers["referrer-policy"] == "strict-origin-when-cross-origin", path


def test_index_supports_head_requests(client):
    # Uptime monitors and link-preview bots often probe with HEAD.
    assert client.head("/").status_code == 200
