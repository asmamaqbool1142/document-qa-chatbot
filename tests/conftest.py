"""Shared fixtures. Every test gets its own temporary data directory, so the
real ``data/`` folder is never touched, and no network or API key is needed."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.core.config import EmbeddingProvider, LLMProvider, Settings
from app.generation.base import ProviderStatus
from app.main import create_app
from app.models.domain import ContextSource, GeneratedAnswer
from app.retrieval.embeddings import HashingEmbedder
from app.services.container import Container, build_container

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"


class FakeGenerator:
    """Records what it was asked and returns a canned, citing answer."""

    name = "fake"
    model = "fake-model"
    is_external = False

    def __init__(self, reply: str | None = None, insufficient: bool = False) -> None:
        self.reply = reply
        self.insufficient = insufficient
        self.calls: list[tuple[str, list[ContextSource]]] = []

    def check(self) -> ProviderStatus:
        return ProviderStatus(True, "fake ok")

    def generate(self, question: str, sources: list[ContextSource]) -> GeneratedAnswer:
        self.calls.append((question, sources))
        if self.insufficient:
            return GeneratedAnswer("", True, self.name, self.model)
        text = self.reply if self.reply is not None else f"Answer from {sources[0].label} [S1]."
        return GeneratedAnswer(text, False, self.name, self.model)


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "data_dir": tmp_path / "data",
        "embedding_provider": EmbeddingProvider.HASHING,
        "llm_provider": LLMProvider.EXTRACTIVE,
        "chunk_size": 400,
        "chunk_overlap": 60,
        "log_level": "WARNING",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


def make_pdf(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=11)
    data: bytes = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def fake_generator() -> FakeGenerator:
    return FakeGenerator()


@pytest.fixture
def container(settings: Settings, fake_generator: FakeGenerator) -> Container:
    return build_container(settings, embedder=HashingEmbedder(), generator=fake_generator)


@pytest.fixture
def client(settings: Settings, fake_generator: FakeGenerator) -> Iterator[TestClient]:
    app = create_app(
        settings,
        container_factory=lambda s: build_container(
            s, embedder=HashingEmbedder(), generator=fake_generator
        ),
    )
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def handbook_bytes() -> bytes:
    return (SAMPLES / "northwind_employee_handbook.txt").read_bytes()


@pytest.fixture
def manual_pdf_bytes() -> bytes:
    return (SAMPLES / "aquapure_w300_manual.pdf").read_bytes()
