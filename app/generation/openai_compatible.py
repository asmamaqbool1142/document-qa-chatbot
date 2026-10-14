"""Hosted answer generation through any OpenAI-compatible Chat Completions API.

Works with OpenAI itself and with compatible services or servers (Groq,
OpenRouter, Together, LM Studio, vLLM, ...) by changing OPENAI_BASE_URL.

Privacy note: with a hosted service, the question and the retrieved document
excerpts are sent to that service. The /health endpoint reports this as
``llm_is_external`` and the frontend shows a notice.
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse

import httpx

from app.core.errors import ConfigurationError, ProviderUnavailableError
from app.generation.base import ChatLLMGenerator, ProviderStatus

logger = logging.getLogger(__name__)

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "host.docker.internal"}


class OpenAICompatibleGenerator(ChatLLMGenerator):
    name = "openai_compatible"

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 600,
        timeout: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        super().__init__(model)
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = client or httpx.Client(timeout=timeout)
        host = urlparse(self._base_url).hostname or ""
        self.is_external = host not in _LOCAL_HOSTS

    def _require_key(self) -> str:
        if not self._api_key:
            raise ConfigurationError(
                "LLM_PROVIDER=openai_compatible requires OPENAI_API_KEY to be set "
                "(in your environment or .env file)."
            )
        return self._api_key

    def complete(self, messages: list[dict[str, str]]) -> str:
        key = self._require_key()
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        try:
            resp = self._client.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {key}"},
            )
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(
                f"Could not reach the LLM API at {self._base_url} ({type(exc).__name__})."
            ) from exc
        if resp.status_code in (401, 403):
            raise ConfigurationError("The LLM API rejected the API key (check OPENAI_API_KEY).")
        if resp.status_code == 404:
            raise ConfigurationError(
                "The LLM API returned 404; check OPENAI_BASE_URL and "
                f"OPENAI_MODEL ('{self.model}')."
            )
        if resp.status_code == 429:
            raise ProviderUnavailableError("The LLM API rate limit or quota was exceeded.")
        if resp.status_code >= 400:
            logger.warning("LLM API returned HTTP %s", resp.status_code)
            raise ProviderUnavailableError(
                f"The LLM API returned an error (HTTP {resp.status_code})."
            )
        try:
            content = resp.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderUnavailableError("The LLM API returned an unexpected response.") from exc
        return str(content or "")

    def check(self) -> ProviderStatus:
        # Configuration check only: we don't spend an API call on every health probe.
        if not self._api_key:
            return ProviderStatus(False, "OPENAI_API_KEY is not set")
        where = "external service" if self.is_external else "local server"
        return ProviderStatus(
            True, f"Configured: model '{self.model}' via {where} {self._base_url}"
        )
