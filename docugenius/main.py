"""FastAPI application factory and ASGI entry point (``docugenius.main:app``)."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from docugenius import __version__
from docugenius.api import api, router
from docugenius.config import FRONTEND_DIR, Settings, get_settings
from docugenius.errors import AppError
from docugenius.services import Services

logger = logging.getLogger("docugenius")

DESCRIPTION = """
DocuGenius is an AI document assistant built on **Retrieval-Augmented Generation**.

* **Documents** - upload a PDF, ask grounded questions with page citations, summarise it, extract tables
* **Assistant** - multi-turn chat, translation and web search powered by Google Gemini
* **Audio** - speech-to-text transcription
* **Social** - Reddit sentiment analysis with charts and AI insights
* **Writing** - style transfer (formal, academic, legal, casual, ...) with an offline fallback engine
* **Sketch** - turn rough strokes into clean lines, circles and rectangles
"""


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for noisy in ("httpx", "httpx2", "google_genai", "chromadb.telemetry", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    settings.documents_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(
        title="DocuGenius API",
        version=__version__,
        description=DESCRIPTION,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.state.settings = settings
    app.state.services = services or Services.build(settings)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):  # type: ignore[no-untyped-def]
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault("Permissions-Policy", "camera=(), geolocation=(), microphone=()")
        return response

    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error. Please try again."})

    app.include_router(router)
    app.include_router(api)

    if FRONTEND_DIR.exists():
        app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")

        @app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(FRONTEND_DIR / "index.html")

    if not settings.google_api_key:
        logger.warning("GOOGLE_API_KEY is not set: AI features are disabled until it is configured.")
    logger.info("DocuGenius %s ready (models: %s)", __version__, ", ".join(settings.gemini_models))
    return app


app = create_app()
