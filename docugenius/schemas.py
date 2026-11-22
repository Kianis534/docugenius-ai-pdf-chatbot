"""Request and response models for the public HTTP API."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, StringConstraints

from docugenius.services.ghostwriter import Engine, Style


def _text(min_length: int, max_length: int) -> Any:
    """A string that is stripped first, so whitespace-only input is rejected."""
    return Annotated[str, StringConstraints(strip_whitespace=True, min_length=min_length, max_length=max_length)]


# --- Shared ---------------------------------------------------------------- #
class SourceOut(BaseModel):
    page: int | None = None
    snippet: str


class WebSourceOut(BaseModel):
    title: str
    url: str


class ErrorOut(BaseModel):
    detail: str


# --- Documents ------------------------------------------------------------- #
class DocumentOut(BaseModel):
    doc_id: str
    filename: str
    pages: int
    chunks: int
    created_at: str


class QuestionIn(BaseModel):
    question: _text(1, 2000) = Field(examples=["What are the key findings?"])


class AnswerOut(BaseModel):
    answer: str
    sources: list[SourceOut] = []
    model: str | None = None


class SummaryOut(BaseModel):
    summary: str
    model: str | None = None


class TableOut(BaseModel):
    page: int
    rows: list[list[str]]
    markdown: str


class TablesOut(BaseModel):
    tables: list[TableOut]


# --- Chat / translate / search -------------------------------------------- #
class ChatTurnIn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatIn(BaseModel):
    message: _text(1, 2000) = Field(examples=["What can you help me with?"])
    doc_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    history: list[ChatTurnIn] = Field(default_factory=list, max_length=20)


class ChatOut(BaseModel):
    reply: str
    mode: Literal["document", "general"]
    sources: list[SourceOut] = []
    model: str | None = None


class TranslateIn(BaseModel):
    text: _text(1, 10_000) = Field(examples=["Knowledge is power."])
    target_language: _text(2, 50) = Field(examples=["Urdu"])


class TranslateOut(BaseModel):
    translation: str
    target_language: str
    model: str | None = None


class SearchIn(BaseModel):
    query: _text(2, 500) = Field(examples=["Latest developments in retrieval-augmented generation"])


class SearchOut(BaseModel):
    answer: str
    grounded: bool = Field(description="True when the answer was grounded with live Google Search results.")
    sources: list[WebSourceOut] = []
    model: str | None = None


# --- Audio ----------------------------------------------------------------- #
class TranscriptionOut(BaseModel):
    text: str
    language: str


# --- Reddit ---------------------------------------------------------------- #
class RedditIn(BaseModel):
    subject: _text(2, 100) = Field(examples=["artificial intelligence"])
    time_filter: Literal["hour", "day", "week", "month", "year", "all"] = "week"
    limit: int = Field(default=25, ge=5, le=100)


class RedditPostOut(BaseModel):
    text: str
    user: str
    upvotes: int
    subreddit: str
    url: str
    created_at: str
    is_submission: bool
    sentiment: Literal["positive", "neutral", "negative"]
    polarity: float


class RedditOut(BaseModel):
    subject: str
    time_filter: str
    total_posts: int
    submissions: int
    comments: int
    positive: float
    neutral: float
    negative: float
    average_polarity: float
    average_upvotes: float
    overall: str
    top_subreddits: dict[str, int]
    chart: str | None = Field(default=None, description="PNG donut chart as a data URI.")
    sample_posts: list[RedditPostOut]
    analysis: str
    analysis_engine: Literal["gemini", "heuristic"]


# --- Ghostwriter ----------------------------------------------------------- #
class GhostwriteIn(BaseModel):
    text: _text(1, 5000) = Field(examples=["We need to fix this problem quickly because it is important."])
    style: Style = "formal"
    engine: Engine = Field(default="auto", description="'auto' uses Gemini when available, otherwise rules.")


class GhostwriteOut(BaseModel):
    original: str
    rewritten: str
    style: Style
    engine: Literal["gemini", "rules"]
    warnings: list[str] = []


# --- Sketch ---------------------------------------------------------------- #
class PointIn(BaseModel):
    x: float
    y: float


class StrokeIn(BaseModel):
    points: list[PointIn] = Field(min_length=1, max_length=5000)


class SketchIn(BaseModel):
    strokes: list[StrokeIn] = Field(min_length=1, max_length=200)


class EnhancedStrokeOut(BaseModel):
    type: Literal["line", "circle", "rectangle", "curve"]
    points: list[PointIn]


class SketchOut(BaseModel):
    strokes: list[EnhancedStrokeOut]
    summary: dict[str, int]
    message: str


# --- Health ---------------------------------------------------------------- #
class HealthOut(BaseModel):
    status: Literal["ok"]
    version: str
    services: dict[str, dict[str, Any]]
    limits: dict[str, int] = Field(default_factory=dict, description="Upload size limits in megabytes.")
