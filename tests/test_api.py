"""HTTP API behaviour via FastAPI's TestClient."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import LLMProvider
from app.core.errors import ProviderUnavailableError
from app.main import create_app
from app.services.container import build_container
from tests.conftest import make_pdf, make_settings


def _upload(client: TestClient, name: str, data: bytes, mime: str = "text/plain"):
    return client.post("/documents/upload", files={"file": (name, data, mime)})


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok" and body["document_count"] == 0
    assert body["embedding"]["detail"].startswith("hashing")
    assert resp.headers["X-Request-ID"]


def test_openapi_docs_available(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    for p in ("/health", "/documents/upload", "/documents", "/documents/{document_id}", "/chat"):
        assert p in paths


def test_upload_list_get_delete_flow(client: TestClient, handbook_bytes: bytes) -> None:
    resp = _upload(client, "handbook.txt", handbook_bytes)
    assert resp.status_code == 201, resp.text
    doc = resp.json()
    assert doc["file_type"] == "txt" and doc["chunk_count"] > 0

    listed = client.get("/documents").json()
    assert listed["total"] == 1 and listed["documents"][0]["document_id"] == doc["document_id"]
    assert client.get(f"/documents/{doc['document_id']}").json()["filename"] == "handbook.txt"

    deleted = client.delete(f"/documents/{doc['document_id']}")
    assert deleted.status_code == 200 and deleted.json()["deleted_chunks"] == doc["chunk_count"]
    assert client.get("/documents").json()["total"] == 0
    assert client.delete(f"/documents/{doc['document_id']}").status_code == 404


def test_upload_pdf(client: TestClient, manual_pdf_bytes: bytes) -> None:
    resp = _upload(client, "manual.pdf", manual_pdf_bytes, "application/pdf")
    assert resp.status_code == 201 and resp.json()["page_count"] == 5


def test_duplicate_upload_returns_409_with_existing_id(
    client: TestClient, handbook_bytes: bytes
) -> None:
    first = _upload(client, "a.txt", handbook_bytes).json()
    dup = _upload(client, "b.txt", handbook_bytes)
    assert dup.status_code == 409
    assert dup.json()["error"] == "duplicate_document"
    assert dup.json()["document_id"] == first["document_id"]


def test_upload_rejections(client: TestClient) -> None:
    cases = [
        (("image.png", b"\x89PNG....", "image/png"), 415, "unsupported_file_type"),
        (("empty.txt", b"", "text/plain"), 422, "invalid_document"),
        (("fake.pdf", b"not a pdf at all", "application/pdf"), 422, "invalid_document"),
        (("blank.txt", b"   \n  ", "text/plain"), 422, "no_extractable_text"),
        (("scan.pdf", make_pdf([""]), "application/pdf"), 422, "no_extractable_text"),
        (("bin.txt", b"\x00\x01\x02binary", "text/plain"), 422, "invalid_document"),
    ]
    for file, status, code in cases:
        resp = client.post("/documents/upload", files={"file": file})
        assert resp.status_code == status, (file[0], resp.text)
        assert resp.json()["error"] == code
        assert "Traceback" not in resp.text


def test_upload_requires_file(client: TestClient) -> None:
    resp = client.post("/documents/upload")
    assert resp.status_code == 422 and resp.json()["error"] == "validation_error"


def test_upload_too_large(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, max_upload_mb=0.001)  # ~1 KB
    with TestClient(create_app(settings)) as client:
        resp = _upload(client, "big.txt", b"word " * 2000)
    assert resp.status_code == 413 and resp.json()["error"] == "file_too_large"


def test_invalid_document_id_format(client: TestClient) -> None:
    assert client.delete("/documents/../../etc").status_code == 404
    resp = client.delete("/documents/not-a-real-id")
    assert resp.status_code == 422


def test_chat_answer_with_citations(client: TestClient, handbook_bytes: bytes) -> None:
    _upload(client, "handbook.txt", handbook_bytes)
    resp = client.post("/chat", json={"question": "How many days of annual leave do I get?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "answered" and body["provider"] == "fake"
    cit = body["citations"][0]
    assert cit["filename"] == "handbook.txt" and cit["page_number"] is None
    assert cit["label"] == "S1" and cit["excerpt"]


def test_chat_pdf_citation_has_page_number(client: TestClient, manual_pdf_bytes: bytes) -> None:
    _upload(client, "manual.pdf", manual_pdf_bytes, "application/pdf")
    body = client.post("/chat", json={"question": "What do error codes E1 and E2 mean?"}).json()
    assert body["citations"][0]["page_number"] == 4


def test_chat_insufficient_evidence(client: TestClient, handbook_bytes: bytes) -> None:
    _upload(client, "handbook.txt", handbook_bytes)
    body = client.post("/chat", json={"question": "Who won the 1966 football World Cup?"}).json()
    assert body["status"] == "insufficient_evidence"
    assert body["answer"].startswith("I couldn't find enough information")
    assert body["citations"] == []


def test_chat_validation(client: TestClient) -> None:
    for payload in (
        {},
        {"question": ""},
        {"question": "  a "},
        {"question": "x" * 2001},
        {"question": "valid question", "top_k": 0},
        {"question": "valid question", "min_score": 5},
    ):
        resp = client.post("/chat", json=payload)
        assert resp.status_code == 422, payload
        assert resp.json()["error"] == "validation_error"


def test_chunks_debug_endpoint_disabled_by_default(
    client: TestClient, handbook_bytes: bytes
) -> None:
    doc = _upload(client, "handbook.txt", handbook_bytes).json()
    assert client.get(f"/documents/{doc['document_id']}/chunks").status_code == 404


def test_chunks_debug_endpoint_when_enabled(tmp_path: Path, handbook_bytes: bytes) -> None:
    settings = make_settings(tmp_path, enable_debug_endpoints=True)
    with TestClient(create_app(settings)) as client:
        doc = _upload(client, "handbook.txt", handbook_bytes).json()
        body = client.get(f"/documents/{doc['document_id']}/chunks?limit=2&offset=1").json()
    assert body["total"] == doc["chunk_count"] and len(body["chunks"]) == 2
    assert body["chunks"][0]["chunk_index"] == 1


def test_provider_failure_returns_503_without_secrets(
    tmp_path: Path, handbook_bytes: bytes
) -> None:
    settings = make_settings(
        tmp_path, llm_provider=LLMProvider.OPENAI_COMPATIBLE, openai_api_key=None
    )
    with TestClient(create_app(settings)) as client:
        health = client.get("/health").json()
        assert health["status"] == "degraded" and health["llm"]["ok"] is False
        assert health["llm_is_external"] is True
        _upload(client, "handbook.txt", handbook_bytes)  # indexing still works without an LLM
        resp = client.post("/chat", json={"question": "How many days of annual leave?"})
    assert resp.status_code == 503
    assert resp.json()["error"] == "configuration_error"
    assert "OPENAI_API_KEY" in resp.json()["message"]


def test_embedding_unavailable_keeps_api_up(tmp_path: Path) -> None:
    def broken_embedder(_settings: object) -> None:
        raise ProviderUnavailableError("model download failed")

    def factory(s):
        from unittest.mock import patch

        with patch("app.services.container.build_embedder", side_effect=broken_embedder):
            return build_container(s)

    with TestClient(create_app(make_settings(tmp_path), container_factory=factory)) as client:
        health = client.get("/health").json()
        assert health["status"] == "degraded" and not health["embedding"]["ok"]
        resp = client.get("/documents")
    assert resp.status_code == 503 and "model download failed" in resp.json()["message"]


def test_unexpected_error_hides_details(tmp_path: Path) -> None:
    def factory(s):
        c = build_container(s)

        class Exploding:
            def ask(self, *a: object) -> None:
                raise RuntimeError("secret internal detail")

        c.chat = Exploding()  # type: ignore[assignment]
        return c

    app = create_app(make_settings(tmp_path), container_factory=factory)
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post("/chat", json={"question": "anything here?"})
    assert resp.status_code == 500
    assert "secret internal detail" not in resp.text
    assert resp.json()["error"] == "internal_error" and resp.json()["request_id"]
