"""HTTP routes.

Blocking work (embeddings, LLM calls, PDF parsing) runs in plain ``def``
endpoints, which FastAPI executes in a worker thread pool, so the event loop
stays responsive.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Path, Request, UploadFile

from docugenius import __version__
from docugenius.errors import PayloadTooLargeError, RedditError
from docugenius.schemas import (
    AnswerOut,
    ChatIn,
    ChatOut,
    DocumentOut,
    ErrorOut,
    GhostwriteIn,
    GhostwriteOut,
    HealthOut,
    QuestionIn,
    RedditIn,
    RedditOut,
    SearchIn,
    SearchOut,
    SketchIn,
    SketchOut,
    SummaryOut,
    TablesOut,
    TranscriptionOut,
    TranslateIn,
    TranslateOut,
)
from docugenius.services import Services
from docugenius.services.assistant import ChatTurn
from docugenius.services.sketch import Point, enhance_strokes

REDDIT_TIMEOUT_SECONDS = 75.0
MAX_AUDIO_BYTES = 25 * 1024 * 1024

ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    code: {"model": ErrorOut} for code in (404, 413, 422, 429, 502, 503)
}

router = APIRouter()
api = APIRouter(prefix="/api", responses=ERROR_RESPONSES)


def get_services(request: Request) -> Services:
    return request.app.state.services


ServicesDep = Annotated[Services, Depends(get_services)]
DocIdPath = Annotated[str, Path(pattern=r"^[a-f0-9]{32}$", description="Document id from upload.")]


def _read_upload(upload: UploadFile, limit: int) -> bytes:
    data = upload.file.read(limit + 1)
    if len(data) > limit:
        raise PayloadTooLargeError(f"The file is larger than the {limit // (1024 * 1024)} MB limit.")
    return data


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
@router.get("/health", response_model=HealthOut, tags=["System"], summary="Service health and configuration")
def health(services: ServicesDep) -> HealthOut:
    return HealthOut(
        status="ok",
        version=__version__,
        services={
            "gemini": services.llm.status(),
            "reddit": {"configured": services.reddit.is_configured},
            "documents": {"indexed": services.documents.count()},
            "speech": {"ffmpeg": services.transcriber.ffmpeg_available()},
        },
        limits={
            "max_upload_mb": services.settings.max_upload_mb,
            "max_audio_mb": MAX_AUDIO_BYTES // (1024 * 1024),
        },
    )


# --------------------------------------------------------------------------- #
# Documents (RAG)
# --------------------------------------------------------------------------- #
@api.post(
    "/documents", response_model=DocumentOut, status_code=201, tags=["Documents"], summary="Upload and index a PDF"
)
def upload_document(services: ServicesDep, file: Annotated[UploadFile, File(description="PDF file")]) -> DocumentOut:
    data = _read_upload(file, services.settings.max_upload_bytes)
    return DocumentOut(**asdict(services.documents.ingest(data, file.filename)))


@api.get("/documents/{doc_id}", response_model=DocumentOut, tags=["Documents"], summary="Get document metadata")
def get_document(services: ServicesDep, doc_id: DocIdPath) -> DocumentOut:
    return DocumentOut(**asdict(services.documents.get(doc_id)))


@api.post(
    "/documents/{doc_id}/query",
    response_model=AnswerOut,
    tags=["Documents"],
    summary="Ask a question about a document (RAG)",
)
def query_document(services: ServicesDep, doc_id: DocIdPath, body: QuestionIn) -> AnswerOut:
    result = services.documents.answer(doc_id, body.question)
    return AnswerOut(answer=result.answer, sources=[asdict(s) for s in result.sources], model=result.model)


@api.post("/documents/{doc_id}/summary", response_model=SummaryOut, tags=["Documents"], summary="Summarise a document")
def summarize_document(services: ServicesDep, doc_id: DocIdPath) -> SummaryOut:
    result = services.documents.summarize(doc_id)
    return SummaryOut(summary=result.answer, model=result.model)


@api.get(
    "/documents/{doc_id}/tables", response_model=TablesOut, tags=["Documents"], summary="Extract tables from a document"
)
def document_tables(services: ServicesDep, doc_id: DocIdPath) -> TablesOut:
    return TablesOut(tables=[asdict(t) for t in services.documents.extract_tables(doc_id)])


# --------------------------------------------------------------------------- #
# Chat, translation, search
# --------------------------------------------------------------------------- #
@api.post("/chat", response_model=ChatOut, tags=["Assistant"], summary="Chat, optionally grounded on a document")
def chat(services: ServicesDep, body: ChatIn) -> ChatOut:
    history = [ChatTurn(role=t.role, content=t.content) for t in body.history]
    result = services.assistant.chat(body.message, doc_id=body.doc_id, history=history)
    return ChatOut(
        reply=result.reply, mode=result.mode, sources=[asdict(s) for s in result.sources], model=result.model
    )


@api.post("/translate", response_model=TranslateOut, tags=["Assistant"], summary="Translate text")
def translate(services: ServicesDep, body: TranslateIn) -> TranslateOut:
    translation, model = services.assistant.translate(body.text, body.target_language)
    return TranslateOut(translation=translation, target_language=body.target_language, model=model)


@api.post(
    "/search",
    response_model=SearchOut,
    tags=["Assistant"],
    summary="Answer a question with Google Search grounding when available",
)
def search(services: ServicesDep, body: SearchIn) -> SearchOut:
    result = services.assistant.search(body.query)
    return SearchOut(answer=result.answer, grounded=result.grounded, sources=result.sources, model=result.model)


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #
@api.post("/transcribe", response_model=TranscriptionOut, tags=["Audio"], summary="Transcribe speech to text")
def transcribe(
    services: ServicesDep,
    file: Annotated[UploadFile, File(description="Audio file (WAV works everywhere; MP3/M4A/OGG need FFmpeg)")],
    language: Annotated[str, Form(pattern=r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,4})?$")] = "en-US",
) -> TranscriptionOut:
    data = _read_upload(file, MAX_AUDIO_BYTES)
    return TranscriptionOut(text=services.transcriber.transcribe(data, language=language), language=language)


# --------------------------------------------------------------------------- #
# Reddit sentiment
# --------------------------------------------------------------------------- #
@api.post("/reddit/sentiment", response_model=RedditOut, tags=["Social"], summary="Reddit sentiment analysis")
async def reddit_sentiment(services: ServicesDep, body: RedditIn) -> RedditOut:
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(services.reddit.analyze, body.subject, body.time_filter, body.limit),
            timeout=REDDIT_TIMEOUT_SECONDS,
        )
    except TimeoutError as exc:
        raise RedditError(
            "Reddit took too long to respond. Try a narrower time range or a more specific topic.",
            status_code=504,
        ) from exc
    return RedditOut(**result)


# --------------------------------------------------------------------------- #
# Ghostwriter
# --------------------------------------------------------------------------- #
@api.post("/ghostwriter", response_model=GhostwriteOut, tags=["Writing"], summary="Rewrite text in another style")
def ghostwrite(services: ServicesDep, body: GhostwriteIn) -> GhostwriteOut:
    return GhostwriteOut(**asdict(services.ghostwriter.rewrite(body.text, body.style, engine=body.engine)))


# --------------------------------------------------------------------------- #
# Sketch
# --------------------------------------------------------------------------- #
@api.post("/sketch/enhance", response_model=SketchOut, tags=["Sketch"], summary="Turn rough strokes into clean shapes")
def enhance_sketch(body: SketchIn) -> SketchOut:
    strokes = enhance_strokes([[Point(p.x, p.y) for p in stroke.points] for stroke in body.strokes])
    summary = dict(Counter(s.type for s in strokes))
    described = ", ".join(f"{count} {kind}{'s' if count > 1 else ''}" for kind, count in summary.items())
    return SketchOut(
        strokes=[{"type": s.type, "points": [{"x": p.x, "y": p.y} for p in s.points]} for s in strokes],
        summary=summary,
        message=f"Cleaned up {described}.",
    )
