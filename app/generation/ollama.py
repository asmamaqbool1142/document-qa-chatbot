"""Local answer generation through an Ollama server (https://ollama.com).

Ollama runs models on your own machine, so document excerpts never leave it.
API used: POST /api/chat (non-streaming) and GET /api/tags for health checks.
"""

from __future__ import annotations

import logging

import httpx

from app.core.errors import ProviderUnavailableError
from app.generation.base import ChatLLMGenerator, ProviderStatus

logger = logging.getLogger(__name__)


class OllamaGenerator(ChatLLMGenerator):
    name = "ollama"
    is_external = False

    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 600,
        timeout: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        super().__init__(model)
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = client or httpx.Client(timeout=timeout)

    def complete(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self._temperature, "num_predict": self._max_tokens},
        }
        try:
            resp = self._client.post(f"{self._base_url}/api/chat", json=payload)
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(
                f"Could not reach Ollama at {self._base_url}. Is it running? "
                "Start it with 'ollama serve' and pull the model with "
                f"'ollama pull {self.model}'."
            ) from exc
        if resp.status_code == 404:
            raise ProviderUnavailableError(
                f"Ollama model '{self.model}' is not installed. Run: ollama pull {self.model}"
            )
        if resp.status_code >= 400:
            logger.warning("Ollama returned HTTP %s", resp.status_code)
            raise ProviderUnavailableError(f"Ollama returned an error (HTTP {resp.status_code}).")
        try:
            return str(resp.json()["message"]["content"])
        except (ValueError, KeyError, TypeError) as exc:
            raise ProviderUnavailableError("Ollama returned an unexpected response.") from exc

    def check(self) -> ProviderStatus:
        try:
            resp = self._client.get(f"{self._base_url}/api/tags", timeout=3.0)
            resp.raise_for_status()
            names = {m.get("name", "") for m in resp.json().get("models", [])}
        except (httpx.HTTPError, ValueError) as exc:
            return ProviderStatus(
                False, f"Ollama not reachable at {self._base_url} ({type(exc).__name__})"
            )
        wanted = {self.model, f"{self.model}:latest"}
        if names & wanted:
            return ProviderStatus(True, f"Ollama model '{self.model}' available")
        return ProviderStatus(
            False,
            f"Ollama is running but model '{self.model}' is not pulled "
            f"(run: ollama pull {self.model})",
        )
