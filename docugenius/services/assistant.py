"""General-purpose LLM features: conversational chat, translation and web search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from docugenius.services.documents import DocumentStore, Source
    from docugenius.services.llm import GeminiClient

CHAT_SYSTEM_PROMPT = (
    "You are DocuGenius, a friendly and knowledgeable assistant that helps people work with "
    "documents: reading, summarising, translating, writing and analysing text. Be accurate, "
    "concise and practical. Use Markdown for lists and emphasis. If you are unsure, say so."
)
TRANSLATE_SYSTEM_PROMPT = (
    "You are a professional translator. Translate faithfully, preserving meaning, tone, "
    "formatting, names and numbers. Reply with the translation only."
)
SEARCH_SYSTEM_PROMPT = (
    "You are a research assistant. Give a direct, well-structured answer with the most "
    "relevant facts first. Use Markdown bullet points where helpful and mention dates for "
    "time-sensitive facts."
)
MAX_HISTORY_TURNS = 8


@dataclass
class ChatTurn:
    role: Literal["user", "assistant"]
    content: str


@dataclass
class ChatReply:
    reply: str
    mode: Literal["document", "general"]
    sources: list[Source] = field(default_factory=list)
    model: str | None = None


@dataclass
class SearchResult:
    answer: str
    grounded: bool
    sources: list[dict[str, str]]
    model: str


def format_history(history: list[ChatTurn]) -> str:
    turns = history[-MAX_HISTORY_TURNS:]
    return "\n".join(f"{'User' if t.role == 'user' else 'Assistant'}: {t.content.strip()}" for t in turns)


class Assistant:
    def __init__(self, llm: GeminiClient, documents: DocumentStore) -> None:
        self._llm = llm
        self._documents = documents

    def chat(self, message: str, doc_id: str | None = None, history: list[ChatTurn] | None = None) -> ChatReply:
        transcript = format_history(history or [])
        if doc_id:
            result = self._documents.answer(doc_id, message, history=transcript or None)
            return ChatReply(reply=result.answer, mode="document", sources=result.sources, model=result.model)

        prompt = (f"Conversation so far:\n{transcript}\n\n" if transcript else "") + f"User: {message}"
        response = self._llm.generate(prompt, system_instruction=CHAT_SYSTEM_PROMPT, temperature=0.6)
        return ChatReply(reply=response.text, mode="general", model=response.model)

    def translate(self, text: str, target_language: str) -> tuple[str, str]:
        prompt = f'Translate the following text into {target_language}.\n\nText:\n"""\n{text}\n"""'
        response = self._llm.generate(prompt, system_instruction=TRANSLATE_SYSTEM_PROMPT, temperature=0.2)
        return response.text, response.model

    def search(self, query: str) -> SearchResult:
        response = self._llm.generate(query, system_instruction=SEARCH_SYSTEM_PROMPT, use_search=True)
        return SearchResult(
            answer=response.text, grounded=response.grounded, sources=response.sources, model=response.model
        )
