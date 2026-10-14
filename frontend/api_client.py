"""Thin HTTP client for the backend. The frontend holds no business logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


class BackendError(Exception):
    """A user-presentable error from the backend or the network."""

    def __init__(
        self, message: str, status_code: int | None = None, payload: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.payload = payload or {}


@dataclass
class ApiClient:
    base_url: str
    timeout: float = 180.0

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}{path}"
        try:
            resp = httpx.request(method, url, timeout=self.timeout, **kwargs)
        except httpx.TimeoutException as exc:
            raise BackendError("The backend took too long to respond.") from exc
        except httpx.HTTPError as exc:
            raise BackendError(
                f"Cannot reach the backend at {self.base_url}. Is the API running?"
            ) from exc
        try:
            payload: dict[str, Any] = resp.json()
        except ValueError:
            payload = {}
        if resp.status_code >= 400:
            message = payload.get("message") or f"Request failed (HTTP {resp.status_code})."
            raise BackendError(str(message), resp.status_code, payload)
        return payload

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def list_documents(self) -> list[dict[str, Any]]:
        return list(self._request("GET", "/documents").get("documents", []))

    def upload(self, filename: str, content: bytes, mime_type: str) -> dict[str, Any]:
        return self._request(
            "POST", "/documents/upload", files={"file": (filename, content, mime_type)}
        )

    def delete(self, document_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/documents/{document_id}")

    def ask(
        self, question: str, top_k: int, min_score: float | None, document_ids: list[str] | None
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"question": question, "top_k": top_k}
        if min_score is not None:
            body["min_score"] = min_score
        if document_ids:
            body["document_ids"] = document_ids
        return self._request("POST", "/chat", json=body)
