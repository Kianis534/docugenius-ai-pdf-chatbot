"""Shared fixtures. External services (Gemini, Reddit, Google Speech) are replaced
with in-memory fakes and embeddings with a deterministic fake, so the suite runs
offline in a few seconds and never downloads a model."""

from __future__ import annotations

import io
import random
import wave
from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding

from docugenius.config import Settings
from docugenius.errors import LLMError, LLMNotConfiguredError
from docugenius.main import create_app
from docugenius.services import Services
from docugenius.services.assistant import Assistant
from docugenius.services.documents import DocumentStore
from docugenius.services.ghostwriter import Ghostwriter, RuleBasedRewriter
from docugenius.services.llm import LLMResponse
from docugenius.services.reddit import RedditSentimentAnalyzer
from docugenius.services.speech import Transcriber

ENV_VARS = (
    "GOOGLE_API_KEY",
    "GEMINI_MODEL",
    "GEMINI_FALLBACK_MODELS",
    "REDDIT_CLIENT_ID",
    "REDDIT_CLIENT_SECRET",
    "DATA_DIR",
    "MAX_UPLOAD_MB",
    "GEMINI_DAILY_REQUEST_LIMIT",
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeLLM:
    """Stands in for GeminiClient and records every prompt it receives."""

    def __init__(self, configured: bool = True, reply: str = "Revenue grew 12% (p. 1).") -> None:
        self.configured = configured
        self.reply = reply
        self.error: LLMError | None = None
        self.grounded = False
        self.sources: list[dict[str, str]] = []
        self.calls: list[dict[str, Any]] = []

    @property
    def is_configured(self) -> bool:
        return self.configured

    @property
    def models(self) -> list[str]:
        return ["fake-model"]

    def generate(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        temperature: float | None = None,
        use_search: bool = False,
    ) -> LLMResponse:
        self.calls.append({"prompt": prompt, "system": system_instruction, "use_search": use_search})
        if not self.configured:
            raise LLMNotConfiguredError()
        if self.error:
            raise self.error
        return LLMResponse(text=self.reply, model="fake-model", grounded=self.grounded, sources=self.sources)

    def status(self) -> dict[str, Any]:
        return {
            "configured": self.configured,
            "models": self.models,
            "cooling_down": [],
            "requests_today": len(self.calls),
            "daily_limit": None,
        }


class FakeRecognizer:
    def __init__(self, transcript: str = "hello from docugenius") -> None:
        self.transcript = transcript
        self.languages: list[str] = []

    def record(self, source: Any) -> str:
        return "audio-data"

    def recognize_google(self, audio: Any, language: str = "en-US") -> str:
        self.languages.append(language)
        return self.transcript


@dataclass
class FakeSubreddit:
    display_name: str


@dataclass
class FakeComment:
    body: str
    score: int
    author: str = "commenter"
    permalink: str = "/r/test/comments/abc/c1"
    created_utc: float = 1_700_000_000.0


class FakeCommentForest(list):
    def replace_more(self, limit: int = 0) -> list:
        return []


@dataclass
class FakeSubmission:
    title: str
    selftext: str
    score: int
    subreddit: FakeSubreddit
    comments: FakeCommentForest = field(default_factory=FakeCommentForest)
    author: str = "poster"
    permalink: str = "/r/test/comments/abc/post"
    created_utc: float = 1_700_000_000.0
    id: str = "abc"
    comment_sort: str = "best"


class FakeReddit:
    def __init__(self, submissions: list[FakeSubmission]) -> None:
        self.submissions = submissions
        self.queries: list[dict[str, Any]] = []
        self.read_only = True

    def subreddit(self, name: str) -> FakeReddit:
        return self

    def search(self, query: str, sort: str, time_filter: str, limit: int):
        self.queries.append({"query": query, "time_filter": time_filter, "limit": limit})
        return iter(self.submissions[:limit])


def sample_submissions() -> list[FakeSubmission]:
    tech, cars = FakeSubreddit("technology"), FakeSubreddit("cars")
    return [
        FakeSubmission(
            "Electric cars are amazing",
            "I love how quiet and fast they are. Great experience!",
            420,
            tech,
            FakeCommentForest(
                [FakeComment("Totally agree, wonderful cars", 50), FakeComment("Charging is terrible and slow", 30)]
            ),
        ),
        FakeSubmission("Battery range is awful in winter", "Really bad, disappointing and frustrating.", 120, cars),
        FakeSubmission("EV sales report for Q3", "Numbers were published today.", 75, cars),
    ]


# --------------------------------------------------------------------------- #
# Document helpers
# --------------------------------------------------------------------------- #
def make_pdf(pages: list[str], table: list[list[str]] | None = None) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buffer = io.BytesIO()
    styles = getSampleStyleSheet()
    story: list[Any] = []
    for index, text in enumerate(pages):
        if index:
            story.append(PageBreak())
        story.append(Paragraph(text, styles["Normal"]))
    if table:
        story.append(Spacer(1, 24))
        grid = Table(table)
        grid.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.75, colors.black)]))
        story.append(grid)
    SimpleDocTemplate(buffer, pagesize=A4).build(story)
    return buffer.getvalue()


def make_blank_pdf() -> bytes:
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def make_wav(seconds: float = 0.5, rate: int = 16_000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\x00\x00" * int(seconds * rate))
    return buffer.getvalue()


REPORT_PAGES = [
    "Annual Report 2024. Revenue grew 12 percent year over year, driven by strong demand in Asia.",
    "Operating costs fell by 3 percent thanks to automation. The company hired 120 engineers.",
    "Outlook: management expects double-digit growth in 2025 and plans to expand into Europe.",
]
REPORT_TABLE = [["Region", "Revenue", "Growth"], ["Asia", "$4.2M", "18%"], ["Europe", "$2.1M", "6%"]]


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(_env_file=None, data_dir=tmp_path / "data", max_upload_mb=2)


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def fake_reddit() -> FakeReddit:
    return FakeReddit(sample_submissions())


@pytest.fixture
def fake_recognizer() -> FakeRecognizer:
    return FakeRecognizer()


@pytest.fixture
def document_store(settings: Settings, fake_llm: FakeLLM) -> DocumentStore:
    return DocumentStore(settings, fake_llm, embeddings=DeterministicFakeEmbedding(size=64))


@pytest.fixture
def services(settings, fake_llm, fake_reddit, fake_recognizer, document_store) -> Services:
    return Services(
        settings=settings,
        llm=fake_llm,
        documents=document_store,
        assistant=Assistant(fake_llm, document_store),
        ghostwriter=Ghostwriter(fake_llm, RuleBasedRewriter(random.Random(7))),
        transcriber=Transcriber(recognizer=fake_recognizer),
        reddit=RedditSentimentAnalyzer(settings, fake_llm, reddit=fake_reddit),
    )


@pytest.fixture
def client(settings: Settings, services: Services):
    with TestClient(create_app(settings, services)) as test_client:
        yield test_client


@pytest.fixture
def report_pdf() -> bytes:
    return make_pdf(REPORT_PAGES, table=REPORT_TABLE)


@pytest.fixture
def uploaded(client: TestClient, report_pdf: bytes) -> dict[str, Any]:
    response = client.post("/api/documents", files={"file": ("report.pdf", report_pdf, "application/pdf")})
    assert response.status_code == 201, response.text
    return response.json()
