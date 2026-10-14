from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import EmbeddingProvider, Settings
from tests.conftest import make_settings


def test_overlap_must_be_smaller_than_chunk_size(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
        make_settings(tmp_path, chunk_size=200, chunk_overlap=200)


def test_default_threshold_depends_on_embedder(tmp_path: Path) -> None:
    st = make_settings(tmp_path, embedding_provider=EmbeddingProvider.SENTENCE_TRANSFORMERS)
    assert st.effective_min_score == 0.30
    assert make_settings(tmp_path).effective_min_score == 0.12
    assert make_settings(tmp_path, min_relevance_score=0.5).effective_min_score == 0.5


def test_settings_read_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CHUNK_SIZE", "750")
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    s = Settings(_env_file=None, data_dir=tmp_path)  # type: ignore[call-arg]
    assert s.chunk_size == 750 and s.llm_provider.value == "openai_compatible"
    assert s.openai_api_key is not None
    assert "sk-from-env" not in repr(s), "secrets must be masked in repr/logs"


def test_invalid_provider_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "magic")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_env_example_parses(tmp_path: Path) -> None:
    """The shipped .env.example must be loadable as-is (empty values mean 'unset')."""
    example = Path(__file__).resolve().parent.parent / ".env.example"
    s = Settings(_env_file=example)  # type: ignore[call-arg]
    assert s.min_relevance_score is None and s.openai_api_key is None
    assert s.effective_min_score == 0.30
