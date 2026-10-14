"""The answer-generator interface and shared plumbing for chat-style LLMs."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Protocol

from app.generation.prompts import INSUFFICIENT_SENTINEL, build_messages
from app.models.domain import ContextSource, GeneratedAnswer


@dataclass(frozen=True)
class ProviderStatus:
    ok: bool
    detail: str


class AnswerGenerator(Protocol):
    """Anything that turns a question plus labelled sources into an answer."""

    name: str
    model: str
    is_external: bool  # True if data leaves this machine/network

    def generate(self, question: str, sources: list[ContextSource]) -> GeneratedAnswer: ...

    def check(self) -> ProviderStatus: ...


class ChatLLMGenerator(abc.ABC):
    """Base for providers that take a list of chat messages and return text."""

    name: str = "llm"
    is_external: bool = False

    def __init__(self, model: str) -> None:
        self.model = model

    @abc.abstractmethod
    def complete(self, messages: list[dict[str, str]]) -> str:
        """Send messages to the model and return the reply text."""

    @abc.abstractmethod
    def check(self) -> ProviderStatus: ...

    def generate(self, question: str, sources: list[ContextSource]) -> GeneratedAnswer:
        reply = self.complete(build_messages(question, sources)).strip()
        insufficient = not reply or _is_insufficient(reply)
        return GeneratedAnswer(
            text=reply, insufficient_evidence=insufficient, provider=self.name, model=self.model
        )


def _is_insufficient(reply: str) -> bool:
    # The prompt asks for the sentinel alone; tolerate light decoration around it.
    cleaned = reply.strip().strip("`*_\"'. ").upper()
    return cleaned == INSUFFICIENT_SENTINEL or (
        INSUFFICIENT_SENTINEL in reply.upper() and len(reply) < 2 * len(INSUFFICIENT_SENTINEL) + 40
    )
