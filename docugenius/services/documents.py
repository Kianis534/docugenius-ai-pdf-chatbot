"""Document ingestion and Retrieval-Augmented Generation (RAG).

Supported formats: PDF, Word (.docx), plain text (.txt) and Markdown (.md).

Pipeline
--------
1. **Ingest** - text is extracted (per page for PDFs, per section of roughly a
   page for other formats), split into overlapping chunks and embedded with a
   Sentence-Transformers model into a per-document Chroma collection.
2. **Retrieve** - the question is embedded and the top-k most similar chunks
   are fetched (cosine similarity).
3. **Generate** - Gemini answers strictly from the retrieved context and the
   response cites the pages or sections it was grounded on.

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
import zipfile
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from docugenius.config import Settings
from docugenius.errors import DocumentNotFoundError, InvalidDocumentError

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

    from docugenius.services.llm import GeminiClient

logger = logging.getLogger(__name__)

# Keep document processing private and quiet: no Chroma telemetry, no tokenizer fork warnings.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DocumentKind = Literal["pdf", "docx", "text"]

DOC_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")
COLLECTION_NAME = "chunks"
SUMMARY_CONTEXT_CHARS = 40_000
MAX_TABLE_PAGES = 100
SECTION_CHARS = 3_000  # roughly one printed page; used as the citation unit for non-PDF files

TEXT_EXTENSIONS = frozenset({".txt", ".md", ".markdown"})
SOURCE_FILES: dict[str, str] = {"pdf": "source.pdf", "docx": "source.docx", "text": "source.txt"}
UNSUPPORTED_MESSAGE = "Unsupported file type. Upload a PDF, Word (.docx), text (.txt) or Markdown (.md) file."
EMPTY_MESSAGES: dict[str, str] = {
    "pdf": "No selectable text was found in this PDF. Scanned documents need OCR before upload.",
    "docx": "This Word document does not contain any text.",
    "text": "This file does not contain any text.",
}

RAG_SYSTEM_PROMPT = (
    "You are DocuGenius, a precise document analyst. Answer using only the provided "
    "context from the user's document. If the context does not contain the answer, say "
    "so plainly instead of guessing. Cite the page or section labels from the context, "
    "like (p. 3) or (§ 2), when you rely on them. Use short paragraphs or bullet points "
    "and Markdown formatting where it helps."
)

SUMMARY_SYSTEM_PROMPT = (
    "You are DocuGenius, an expert at summarising documents faithfully. Never invent facts "
    "that are not present in the provided excerpts."
)


@dataclass
class DocumentInfo:
    doc_id: str
    filename: str
    pages: int  # pages for PDFs, sections for other formats
    chunks: int
    created_at: str
    kind: str = "pdf"  # default keeps indexes created before multi-format support readable


@dataclass
class Source:
    page: int | None
    snippet: str
    section: int | None = None


@dataclass
class RagAnswer:
    answer: str
    sources: list[Source] = field(default_factory=list)
    model: str | None = None


@dataclass
class ExtractedTable:
    page: int | None
    rows: list[list[str]]
    markdown: str


class DocumentStore:
    def __init__(self, settings: Settings, llm: GeminiClient, embeddings: Embeddings | None = None) -> None:
        self._settings = settings
        self._llm = llm
        self._embeddings = embeddings
        self._root = settings.documents_dir
        self._write_lock = threading.Lock()
        self._embeddings_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Embeddings (lazy: the model is ~90 MB and only needed for documents)
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
        filename = Path(filename or "document").name
        kind = detect_kind(data, filename)
        doc_id = hashlib.sha256(data).hexdigest()[:32]
        doc_dir = self._root / doc_id

        with self._write_lock:
            existing = self._load_meta(doc_id)
            if existing:
                logger.info("Document %s already indexed, reusing it", doc_id)
                return replace(existing, filename=filename)

            units, unit_count = self._extract_units(kind, data)
            if not units:
                raise InvalidDocumentError(EMPTY_MESSAGES[kind])

            from langchain_chroma import Chroma
            from langchain_core.documents import Document
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            unit_key = "page" if kind == "pdf" else "section"
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self._settings.chunk_size,
                chunk_overlap=min(self._settings.chunk_overlap, self._settings.chunk_size // 2),
            )
            chunks = splitter.split_documents(
                [Document(page_content=text, metadata={unit_key: number}) for number, text in units]
            )
            for index, chunk in enumerate(chunks):
                chunk.metadata["chunk"] = index

            self._release_index(doc_dir / "index")
            if doc_dir.exists():  # leftover from an interrupted ingest or a deleted document
                shutil.rmtree(doc_dir, ignore_errors=True)
            doc_dir.mkdir(parents=True, exist_ok=True)
            try:
                (doc_dir / SOURCE_FILES[kind]).write_bytes(data)
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
                    pages=unit_count,
                    chunks=len(chunks),
                    created_at=datetime.now(UTC).isoformat(timespec="seconds"),
                    kind=kind,
                )
                (doc_dir / "meta.json").write_text(json.dumps(asdict(info), indent=2), encoding="utf-8")
            except Exception:
                self._release_index(doc_dir / "index")
                shutil.rmtree(doc_dir, ignore_errors=True)
                raise

        logger.info("Indexed %s %s (%s): %d units, %d chunks", kind, filename, doc_id, info.pages, info.chunks)
        return info

    def get(self, doc_id: str) -> DocumentInfo:
        info = self._load_meta(self._validate_id(doc_id))
        if not info:
            raise DocumentNotFoundError()
        return info

    def delete(self, doc_id: str) -> None:
        """Remove a document, its vector index and any cached results from disk."""
        doc_dir = self._doc_dir(doc_id)
        with self._write_lock:
            # Remove the metadata first: even if an index file stays locked for a moment
            # (Windows), the document is gone from the API's point of view.
            (doc_dir / "meta.json").unlink(missing_ok=True)
            self._release_index(doc_dir / "index")
            shutil.rmtree(doc_dir, ignore_errors=True)
            if doc_dir.exists():
                logger.warning("Some files for document %s could not be removed yet", doc_id)
        logger.info("Deleted document %s", doc_id)

    # ------------------------------------------------------------------ #
    # Retrieval-augmented features
    # ------------------------------------------------------------------ #
    def answer(self, doc_id: str, question: str, history: str | None = None) -> RagAnswer:
        store = self._open(doc_id)
        hits = store.similarity_search(question, k=self._settings.retrieval_k)
        hits.sort(key=lambda d: d.metadata.get("chunk", 0))
        context = "\n\n".join(f"[{_location_label(d.metadata)}]\n{d.page_content}" for d in hits)
        prompt = (
            f"Conversation so far:\n{history}\n\n" if history else ""
        ) + f"Context from the document:\n{context}\n\nQuestion: {question}"
        response = self._llm.generate(prompt, system_instruction=RAG_SYSTEM_PROMPT)
        sources = [
            Source(page=d.metadata.get("page"), section=d.metadata.get("section"), snippet=_snippet(d.page_content))
            for d in sorted(hits, key=lambda d: d.metadata.get("page") or d.metadata.get("section") or 0)
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
        context = "\n\n".join(f"[{_location_label(meta or {})}]\n{text}" for text, meta in excerpts)
        prompt = (
            "Summarise the document below for a busy reader.\n"
            "Start with a one-sentence overview, then 4-6 bullet points with the key facts, "
            "findings or arguments, then a short 'Key takeaway' line.\n\n"
            f"Document excerpts (in reading order):\n{context}"
        )
        response = self._llm.generate(prompt, system_instruction=SUMMARY_SYSTEM_PROMPT)
        return RagAnswer(answer=response.text, model=response.model)

    def extract_tables(self, doc_id: str) -> list[ExtractedTable]:
        info = self.get(doc_id)
        doc_dir = self._root / doc_id
        cache = doc_dir / "tables.json"
        if cache.exists():
            return [ExtractedTable(**t) for t in json.loads(cache.read_text(encoding="utf-8"))]

        source = doc_dir / SOURCE_FILES.get(info.kind, "source.pdf")
        if info.kind == "docx":
            _, raw_tables = _docx_content(source.read_bytes())
            tables = [ExtractedTable(page=None, rows=rows, markdown=_to_markdown(rows)) for rows in raw_tables]
        elif info.kind == "text":
            raw_tables = _markdown_tables(_decode_text(source.read_bytes()))
            tables = [ExtractedTable(page=None, rows=rows, markdown=_to_markdown(rows)) for rows in raw_tables]
        else:
            tables = self._pdf_tables(source, doc_id)

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
    def _release_index(index_dir: Path) -> None:
        """Stop Chroma's cached client for an index so its files can be deleted or recreated.

        Chroma keeps one shared client per persist directory for the life of the
        process. Without releasing it, Windows keeps the SQLite file locked and a
        re-created index at the same path would reuse a stale handle.
        """
        try:
            from chromadb.api.shared_system_client import SharedSystemClient
        except Exception:  # pragma: no cover - internal API moved
            return
        target = os.path.normcase(os.path.abspath(index_dir))
        systems = getattr(SharedSystemClient, "_identifier_to_system", {})
        refcounts = getattr(SharedSystemClient, "_identifier_to_refcount", {})
        for identifier in list(systems):
            if os.path.normcase(os.path.abspath(identifier)) != target:
                continue
            system = systems.pop(identifier, None)
            refcounts.pop(identifier, None)
            if system is not None:
                try:
                    system.stop()
                except Exception:
                    logger.debug("Could not stop Chroma client for %s", identifier, exc_info=True)

    def _extract_units(self, kind: str, data: bytes) -> tuple[list[tuple[int, str]], int]:
        """Return numbered text units (pages or sections) and the total unit count."""
        if kind == "pdf":
            return self._extract_pages(data)
        if kind == "docx":
            blocks, _ = _docx_content(data)
        else:
            blocks = re.split(r"\n\s*\n", _decode_text(data))
        sections = _group_into_sections(blocks)
        return sections, len(sections)

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

    def _extract_pages(self, data: bytes) -> tuple[list[tuple[int, str]], int]:
        """Return the non-empty page texts and the total page count from a single parse."""
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
        return pages, len(reader.pages)

    @staticmethod
    def _pdf_tables(path: Path, doc_id: str) -> list[ExtractedTable]:
        import pdfplumber

        tables: list[ExtractedTable] = []
        try:
            with pdfplumber.open(path) as pdf:
                for number, page in enumerate(pdf.pages[:MAX_TABLE_PAGES], start=1):
                    for raw in page.extract_tables():
                        rows = [
                            [_clean_cell(cell) for cell in row]
                            for row in raw
                            if row and any(cell not in (None, "") for cell in row)
                        ]
                        if len(rows) >= 2 and max(len(r) for r in rows) >= 2:
                            tables.append(ExtractedTable(page=number, rows=rows, markdown=_to_markdown(rows)))
        except Exception as exc:  # malformed layouts can break pdfplumber's parser
            logger.warning("Table extraction failed for %s: %s", doc_id, exc)
            raise InvalidDocumentError("Tables could not be read from this PDF's layout.") from exc
        return tables


# --------------------------------------------------------------------------- #
# Format detection and parsing
# --------------------------------------------------------------------------- #
def detect_kind(data: bytes, filename: str) -> DocumentKind:
    """Identify the format from the file's bytes, using the extension only for plain text."""
    if data.lstrip()[:5].startswith(b"%PDF"):
        return "pdf"
    if data[:4] == b"PK\x03\x04":
        if _is_docx(data):
            return "docx"
        raise InvalidDocumentError(UNSUPPORTED_MESSAGE)
    if Path(filename).suffix.lower() in TEXT_EXTENSIONS:
        return "text"
    raise InvalidDocumentError(UNSUPPORTED_MESSAGE)


def _is_docx(data: bytes) -> bool:
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            return "word/document.xml" in archive.namelist()
    except zipfile.BadZipFile:
        return False


def _decode_text(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = data.decode("utf-16", errors="replace")
    else:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = data.decode("cp1252", errors="replace")
    if "\x00" in text:
        raise InvalidDocumentError("This file does not look like plain text.")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _docx_content(data: bytes) -> tuple[list[str], list[list[list[str]]]]:
    """Return text blocks (paragraphs, then tables as Markdown) and the parsed tables."""
    import docx

    try:
        document = docx.Document(BytesIO(data))
    except Exception as exc:
        raise InvalidDocumentError("This Word file could not be read. Re-save it as .docx and try again.") from exc

    blocks = [p.text for p in document.paragraphs if p.text.strip()]
    tables: list[list[list[str]]] = []
    for table in document.tables:
        rows = [[_clean_cell(cell.text) for cell in row.cells] for row in table.rows]
        rows = [row for row in rows if any(row)]
        if len(rows) >= 2 and max(len(r) for r in rows) >= 2:
            tables.append(rows)
    # Tables are searchable too: their Markdown form is indexed alongside the paragraphs.
    blocks.extend(_to_markdown(rows) for rows in tables)
    return blocks, tables


_MD_SEPARATOR = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?$")


def _markdown_tables(text: str) -> list[list[list[str]]]:
    """Find GitHub-style pipe tables (header row, separator row, body rows)."""
    tables: list[list[list[str]]] = []
    block: list[str] = []
    for line in [*text.splitlines(), ""]:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.count("|") >= 2:
            block.append(stripped)
            continue
        if len(block) >= 3 and _MD_SEPARATOR.match(block[1]):
            rows = [[cell.strip() for cell in row.strip("|").split("|")] for i, row in enumerate(block) if i != 1]
            if max(len(r) for r in rows) >= 2:
                tables.append(rows)
        block = []
    return tables


def _group_into_sections(blocks: list[str], size: int = SECTION_CHARS) -> list[tuple[int, str]]:
    """Merge paragraphs into numbered sections of roughly ``size`` characters."""
    sections: list[str] = []
    current: list[str] = []
    length = 0
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if current and length + len(block) > size:
            sections.append("\n\n".join(current))
            current, length = [], 0
        current.append(block)
        length += len(block) + 2
    if current:
        sections.append("\n\n".join(current))
    return list(enumerate(sections, start=1))


def _location_label(metadata: dict[str, Any]) -> str:
    if metadata.get("page") is not None:
        return f"Page {metadata['page']}"
    if metadata.get("section") is not None:
        return f"Section {metadata['section']}"
    return "Excerpt"


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
