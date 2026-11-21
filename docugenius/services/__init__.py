"""Service layer. Each service is framework-agnostic and individually testable."""

from __future__ import annotations

from dataclasses import dataclass

from docugenius.config import Settings
from docugenius.services.assistant import Assistant
from docugenius.services.documents import DocumentStore
from docugenius.services.ghostwriter import Ghostwriter
from docugenius.services.llm import GeminiClient
from docugenius.services.reddit import RedditSentimentAnalyzer
from docugenius.services.speech import Transcriber


@dataclass
class Services:
    settings: Settings
    llm: GeminiClient
    documents: DocumentStore
    assistant: Assistant
    ghostwriter: Ghostwriter
    transcriber: Transcriber
    reddit: RedditSentimentAnalyzer

    @classmethod
    def build(cls, settings: Settings) -> Services:
        llm = GeminiClient(settings)
        documents = DocumentStore(settings, llm)
        return cls(
            settings=settings,
            llm=llm,
            documents=documents,
            assistant=Assistant(llm, documents),
            ghostwriter=Ghostwriter(llm),
            transcriber=Transcriber(),
            reddit=RedditSentimentAnalyzer(settings, llm),
        )


__all__ = ["Services"]
