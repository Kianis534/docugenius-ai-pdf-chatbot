"""PDF ingestion and Retrieval-Augmented Generation (RAG).

Pipeline
--------
1. **Ingest** - text is extracted page by page with ``pypdf``, split into
   overlapping chunks and embedded with a Sentence-Transformers model into a
   per-document Chroma collection persisted on disk.
2. **Retrieve** - the question is embedded and the top-k most similar chunks
   are fetched (cosine similarity).
3. **Generate** - Gemini answers strictly from the retrieved context and the
   response cites the pages it was grounded on.

Documents are content-addressed (SHA-256 of the bytes), so re-uploading the
same file is instant and never duplicates work.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import threading
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

from docugenius.config import Settings
from docugenius.errors import DocumentNotFoundError, InvalidDocumentError

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

    from docugenius.services.llm import GeminiClient

logger = logging.getLogger(__name__)

# Keep document processing private and quiet: no Chroma telemetry, no tokenizer fork warnings.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DOC_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")
COLLECTION_NAME = "chunks"
SUMMARY_CONTEXT_CHARS = 40_000
MAX_TABLE_PAGES = 100

RAG_SYSTEM_PROMPT = (
    "You are DocuGenius, a precise document analyst. Answer using only the provided "
    "context from the user's document. If the context does not contain the answer, say "
    "so plainly instead of guessing. Cite page numbers like (p. 3) when you rely on them. "
    "Use short paragraphs or bullet points and Markdown formatting where it helps."
)

SUMMARY_SYSTEM_PROMPT = (
    "You are DocuGenius, an expert at summarising documents faithfully. Never invent facts "
    "that are not present in the provided excerpts."
)


@dataclass
class DocumentInfo:
    doc_id: str
    filename: str
    pages: int
    chunks: int
    created_at: str


@dataclass
class Source:
    page: int | None
    snippet: str


@dataclass
class RagAnswer:
    answer: str
    sources: list[Source] = field(default_factory=list)
    model: str | None = None


@dataclass
class ExtractedTable:
    page: int
    rows: list[list[str]]
    markdown: str


class DocumentStore:
    def __init__(self, settings: Settings, llm: GeminiClient, embeddings: Embeddings | None = None) -> None:
        self._settings = settings
        self._llm = llm
        self._embeddings = embeddings
        self._root = settings.documents_dir
        self._ingest_lock = threading.Lock()
        self._embeddings_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Embeddings (lazy: the model is ~90 MB and only needed for PDFs)
    # ------------------------------------------------------------------ #
    @property
    def embeddings(self) -> Embeddings:
        if self._embeddings is None:
            with self._embeddings_lock:
                if self._embeddings is None:
                    from langchain_huggingface import HuggingFaceEmbeddings

                    logger.info("Loading embedding model %s", self._settings.embedding_model)
                    self._embeddings = HuggingFaceEmbeddings(
                        model_name=self._settings.embedding_model,
                        encode_kwargs={"normalize_embeddings": True},
                    )
        return self._embeddings

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #
    def ingest(self, data: bytes, filename: str | None = None) -> DocumentInfo:
        if not data.lstrip()[:5].startswith(b"%PDF"):
            raise InvalidDocumentError("Only PDF files are supported. Please upload a valid .pdf file.")

        doc_id = hashlib.sha256(data).hexdigest()[:32]
        filename = Path(filename or "document.pdf").name
        doc_dir = self._root / doc_id

        with self._ingest_lock:
            existing = self._load_meta(doc_id)
            if existing:
                logger.info("Document %s already indexed, reusing it", doc_id)
                return replace(existing, filename=filename)

            pages = self._extract_pages(data)
            if not pages:
                raise InvalidDocumentError(
                    "No selectable text was found in this PDF. Scanned documents need OCR before upload."
                )

            from langchain_chroma import Chroma
            from langchain_core.documents import Document
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self._settings.chunk_size,
                chunk_overlap=min(self._settings.chunk_overlap, self._settings.chunk_size // 2),
            )
            chunks = splitter.split_documents(
                [Document(page_content=text, metadata={"page": number}) for number, text in pages]
            )
            for index, chunk in enumerate(chunks):
                chunk.metadata["chunk"] = index

            if doc_dir.exists():  # leftover from an interrupted ingest
                shutil.rmtree(doc_dir, ignore_errors=True)
            doc_dir.mkdir(parents=True, exist_ok=True)
            try:
                (doc_dir / "source.pdf").write_bytes(data)
                Chroma.from_documents(
                    documents=chunks,
                    embedding=self.embeddings,
                    collection_name=COLLECTION_NAME,
                    persist_directory=str(doc_dir / "index"),
                    collection_metadata={"hnsw:space": "cosine"},
                )
                info = DocumentInfo(
                    doc_id=doc_id,
                    filename=filename,
                    pages=self._page_count(data),
                    chunks=len(chunks),
                    created_at=datetime.now(UTC).isoformat(timespec="seconds"),
                )
                (doc_dir / "meta.json").write_text(json.dumps(asdict(info), indent=2), encoding="utf-8")
            except Exception:
                shutil.rmtree(doc_dir, ignore_errors=True)
                raise

        logger.info("Indexed %s (%s): %d pages, %d chunks", filename, doc_id, info.pages, info.chunks)
        return info

    def get(self, doc_id: str) -> DocumentInfo:
        info = self._load_meta(self._validate_id(doc_id))
        if not info:
            raise DocumentNotFoundError()
        return info

    # ------------------------------------------------------------------ #
    # Retrieval-augmented features
    # ------------------------------------------------------------------ #
    def answer(self, doc_id: str, question: str, history: str | None = None) -> RagAnswer:
        store = self._open(doc_id)
        hits = store.similarity_search(question, k=self._settings.retrieval_k)
        hits.sort(key=lambda d: d.metadata.get("chunk", 0))
        context = "\n\n".join(f"[Page {d.metadata.get('page', '?')}]\n{d.page_content}" for d in hits)
        prompt = (
            f"Conversation so far:\n{history}\n\n" if history else ""
        ) + f"Context from the document:\n{context}\n\nQuestion: {question}"
        response = self._llm.generate(prompt, system_instruction=RAG_SYSTEM_PROMPT)
        sources = [
            Source(page=d.metadata.get("page"), snippet=_snippet(d.page_content))
            for d in sorted(hits, key=lambda d: d.metadata.get("page") or 0)
        ]
        return RagAnswer(answer=response.text, sources=sources, model=response.model)

    def summarize(self, doc_id: str) -> RagAnswer:
        store = self._open(doc_id)
        records = store.get(include=["documents", "metadatas"])
        chunks = sorted(
            zip(records.get("documents") or [], records.get("metadatas") or [], strict=False),
            key=lambda pair: (pair[1] or {}).get("chunk", 0),
        )
        excerpts = _sample_evenly(chunks, SUMMARY_CONTEXT_CHARS)
        context = "\n\n".join(f"[Page {(meta or {}).get('page', '?')}]\n{text}" for text, meta in excerpts)
        prompt = (
            "Summarise the document below for a busy reader.\n"
            "Start with a one-sentence overview, then 4-6 bullet points with the key facts, "
            "findings or arguments, then a short 'Key takeaway' line.\n\n"
            f"Document excerpts (in reading order):\n{context}"
        )
        response = self._llm.generate(prompt, system_instruction=SUMMARY_SYSTEM_PROMPT)
        return RagAnswer(answer=response.text, model=response.model)

    def extract_tables(self, doc_id: str) -> list[ExtractedTable]:
        doc_dir = self._doc_dir(doc_id)
        cache = doc_dir / "tables.json"
        if cache.exists():
            return [ExtractedTable(**t) for t in json.loads(cache.read_text(encoding="utf-8"))]

        import pdfplumber

        tables: list[ExtractedTable] = []
        with pdfplumber.open(doc_dir / "source.pdf") as pdf:
            for number, page in enumerate(pdf.pages[:MAX_TABLE_PAGES], start=1):
                for raw in page.extract_tables():
                    rows = [
                        [_clean_cell(cell) for cell in row]
                        for row in raw
                        if row and any(cell not in (None, "") for cell in row)
                    ]
                    if len(rows) >= 2 and max(len(r) for r in rows) >= 2:
                        tables.append(ExtractedTable(page=number, rows=rows, markdown=_to_markdown(rows)))

        cache.write_text(json.dumps([asdict(t) for t in tables]), encoding="utf-8")
        return tables

    def count(self) -> int:
        if not self._root.exists():
            return 0
        return sum(1 for p in self._root.iterdir() if (p / "meta.json").exists())

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_id(doc_id: str) -> str:
        if not DOC_ID_PATTERN.fullmatch(doc_id or ""):
            raise DocumentNotFoundError("Invalid document id.")
        return doc_id

    def _doc_dir(self, doc_id: str) -> Path:
        doc_dir = self._root / self._validate_id(doc_id)
        if not (doc_dir / "meta.json").exists():
            raise DocumentNotFoundError()
        return doc_dir

    def _open(self, doc_id: str) -> Any:
        from langchain_chroma import Chroma

        doc_dir = self._doc_dir(doc_id)
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=str(doc_dir / "index"),
        )

    def _load_meta(self, doc_id: str) -> DocumentInfo | None:
        meta = self._root / doc_id / "meta.json"
        if not meta.exists():
            return None
        return DocumentInfo(**json.loads(meta.read_text(encoding="utf-8")))

    @staticmethod
    def _open_pdf(data: bytes) -> Any:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError

        try:
            reader = PdfReader(BytesIO(data))
            if reader.is_encrypted and not reader.decrypt(""):
                raise InvalidDocumentError("This PDF is password-protected. Remove the password and try again.")
            return reader
        except InvalidDocumentError:
            raise
        except (PdfReadError, ValueError, OSError) as exc:
            raise InvalidDocumentError("The PDF appears to be corrupted and could not be read.") from exc

    def _extract_pages(self, data: bytes) -> list[tuple[int, str]]:
        reader = self._open_pdf(data)
        pages: list[tuple[int, str]] = []
        for number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:  # a single malformed page should not sink the whole document
                logger.warning("Could not extract text from page %d", number, exc_info=True)
                text = ""
            text = re.sub(r"[ \t]+", " ", text).strip()
            if text:
                pages.append((number, text))
        return pages

    def _page_count(self, data: bytes) -> int:
        return len(self._open_pdf(data).pages)


def _snippet(text: str, limit: int = 240) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"


def _sample_evenly(chunks: list[tuple[str, Any]], budget: int) -> list[tuple[str, Any]]:
    """Pick chunks spread across the whole document so long files are covered end to end."""
    total = sum(len(text) for text, _ in chunks)
    if total <= budget:
        return chunks
    keep = max(1, int(len(chunks) * budget / total))
    step = len(chunks) / keep
    return [chunks[int(i * step)] for i in range(keep)]


def _clean_cell(cell: Any) -> str:
    return " ".join(str(cell).split()) if cell is not None else ""


def _to_markdown(rows: list[list[str]]) -> str:
    width = max(len(r) for r in rows)
    padded = [r + [""] * (width - len(r)) for r in rows]

    def line(cells: list[str]) -> str:
        return "| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |"

    header, *body = padded
    return "\n".join([line(header), "| " + " | ".join(["---"] * width) + " |", *map(line, body)])
