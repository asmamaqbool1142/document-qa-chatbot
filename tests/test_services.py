"""Ingestion and chat services working together (no HTTP)."""

from __future__ import annotations

import pytest

from app.core.errors import DocumentNotFoundError, DuplicateDocumentError
from app.models.domain import AnswerStatus
from app.retrieval.embeddings import HashingEmbedder
from app.services.chat_service import NO_EVIDENCE_MESSAGE
from app.services.container import Container, build_container
from app.services.ingestion_service import compute_document_id
from tests.conftest import FakeGenerator, make_settings


def test_ingest_txt_and_list(container: Container, handbook_bytes: bytes) -> None:
    assert container.ingestion is not None and container.store is not None
    record, warnings = container.ingestion.ingest("handbook.txt", handbook_bytes)
    assert warnings == []
    assert record.document_id.startswith("doc_") and len(record.document_id) == 20
    assert record.chunk_count > 3
    assert container.store.count_for_document(record.document_id) == record.chunk_count
    assert [d.document_id for d in container.ingestion.list_documents()] == [record.document_id]


def test_document_id_is_stable_across_instances(tmp_path, handbook_bytes: bytes) -> None:
    a = build_container(make_settings(tmp_path / "a"), HashingEmbedder(), FakeGenerator())
    b = build_container(make_settings(tmp_path / "b"), HashingEmbedder(), FakeGenerator())
    assert a.ingestion and b.ingestion
    id_a = a.ingestion.ingest("x.txt", handbook_bytes)[0].document_id
    id_b = b.ingestion.ingest("renamed.txt", handbook_bytes)[0].document_id
    assert id_a == id_b


def test_duplicate_upload_is_rejected_even_with_new_name(
    container: Container, handbook_bytes: bytes
) -> None:
    assert container.ingestion is not None
    record, _ = container.ingestion.ingest("handbook.txt", handbook_bytes)
    with pytest.raises(DuplicateDocumentError) as info:
        container.ingestion.ingest("copy-of-handbook.txt", handbook_bytes)
    assert info.value.document_id == record.document_id
    assert len(container.ingestion.list_documents()) == 1


def test_delete_removes_vectors_registry_and_stored_file(tmp_path, handbook_bytes: bytes) -> None:
    settings = make_settings(tmp_path, store_uploads=True)
    c = build_container(settings, HashingEmbedder(), FakeGenerator())
    assert c.ingestion and c.store
    record, _ = c.ingestion.ingest("../../handbook.txt", handbook_bytes)
    stored = settings.uploads_dir / f"{record.document_id}.txt"
    assert stored.exists(), "original kept under an internal name"
    assert list(settings.uploads_dir.iterdir()) == [stored]

    assert c.ingestion.delete(record.document_id) == record.chunk_count
    assert not stored.exists()
    assert c.store.count() == 0 and c.ingestion.list_documents() == []
    with pytest.raises(DocumentNotFoundError):
        c.ingestion.delete(record.document_id)
    # After deletion the same file can be uploaded again.
    c.ingestion.ingest("handbook.txt", handbook_bytes)


def test_uploads_not_stored_by_default(container: Container, handbook_bytes: bytes) -> None:
    assert container.ingestion is not None
    container.ingestion.ingest("handbook.txt", handbook_bytes)
    assert list(container.settings.uploads_dir.iterdir()) == []


def test_failed_indexing_rolls_back_vectors(
    container: Container, handbook_bytes: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert container.ingestion and container.registry and container.store

    def boom(*_args: object) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(container.registry, "add", boom)
    with pytest.raises(RuntimeError):
        container.ingestion.ingest("handbook.txt", handbook_bytes)
    assert container.store.count() == 0


def test_pdf_chunks_carry_page_numbers(container: Container, manual_pdf_bytes: bytes) -> None:
    assert container.ingestion and container.store
    record, _ = container.ingestion.ingest("manual.pdf", manual_pdf_bytes)
    rows = container.store.get_document_chunks(record.document_id, limit=100, offset=0)
    assert {page for _, _, page, _ in rows} == {1, 2, 3, 4, 5}
    assert compute_document_id(record.content_hash) == record.document_id


def test_chat_relevant_question_returns_citations(
    container: Container, fake_generator: FakeGenerator, handbook_bytes: bytes
) -> None:
    assert container.ingestion and container.chat
    container.ingestion.ingest("handbook.txt", handbook_bytes)
    result = container.chat.ask("How many weeks of parental leave do birth parents get?")
    assert result.status is AnswerStatus.ANSWERED
    assert result.citations and result.citations[0].filename == "handbook.txt"
    assert result.citations[0].explicitly_cited
    # The generator saw the parental-leave passage.
    _, sources = fake_generator.calls[0]
    assert any("16 weeks" in s.text for s in sources)


def test_chat_irrelevant_question_skips_llm(
    container: Container, fake_generator: FakeGenerator, handbook_bytes: bytes
) -> None:
    assert container.ingestion and container.chat
    container.ingestion.ingest("handbook.txt", handbook_bytes)
    result = container.chat.ask("What is the boiling point of tungsten on Mars?")
    assert result.status is AnswerStatus.INSUFFICIENT_EVIDENCE
    assert result.answer == NO_EVIDENCE_MESSAGE and result.citations == []
    assert fake_generator.calls == []


def test_chat_with_no_documents(container: Container) -> None:
    assert container.chat
    assert container.chat.ask("anything at all?").status is AnswerStatus.INSUFFICIENT_EVIDENCE


def test_chat_llm_reports_insufficient_evidence(tmp_path, handbook_bytes: bytes) -> None:
    c = build_container(
        make_settings(tmp_path), HashingEmbedder(), FakeGenerator(insufficient=True)
    )
    assert c.ingestion and c.chat
    c.ingestion.ingest("handbook.txt", handbook_bytes)
    result = c.chat.ask("How many days of annual leave do employees get?")
    assert result.status is AnswerStatus.INSUFFICIENT_EVIDENCE
    assert result.citations == []


def test_chat_strips_invented_source_labels(tmp_path, handbook_bytes: bytes) -> None:
    gen = FakeGenerator(reply="Leave is 22 days [S1][S42]. Also see [S9].")
    c = build_container(make_settings(tmp_path), HashingEmbedder(), gen)
    assert c.ingestion and c.chat
    c.ingestion.ingest("handbook.txt", handbook_bytes)
    result = c.chat.ask("How many days of annual leave do employees get?")
    assert "[S42]" not in result.answer and "[S9]" not in result.answer
    assert [cit.label for cit in result.citations] == ["S1"]


def test_chat_without_markers_lists_sources_as_not_cited(tmp_path, handbook_bytes: bytes) -> None:
    gen = FakeGenerator(reply="Employees get 22 days.")
    c = build_container(make_settings(tmp_path), HashingEmbedder(), gen)
    assert c.ingestion and c.chat
    c.ingestion.ingest("handbook.txt", handbook_bytes)
    result = c.chat.ask("How many days of annual leave do employees get?")
    assert result.citations and not any(cit.explicitly_cited for cit in result.citations)


def test_chat_document_filter(
    container: Container, handbook_bytes: bytes, manual_pdf_bytes: bytes
) -> None:
    assert container.ingestion and container.chat
    container.ingestion.ingest("handbook.txt", handbook_bytes)
    manual, _ = container.ingestion.ingest("manual.pdf", manual_pdf_bytes)
    result = container.chat.ask("warranty period", min_score=0.0, document_ids=[manual.document_id])
    assert {c.document_id for c in result.retrieval.candidates} == {manual.document_id}
