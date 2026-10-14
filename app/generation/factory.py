"""Create the configured answer generator from settings."""

from __future__ import annotations

from app.core.config import LLMProvider, Settings
from app.generation.base import AnswerGenerator
from app.generation.extractive import ExtractiveGenerator
from app.generation.ollama import OllamaGenerator
from app.generation.openai_compatible import OpenAICompatibleGenerator


def build_generator(settings: Settings) -> AnswerGenerator:
    if settings.llm_provider is LLMProvider.OLLAMA:
        return OllamaGenerator(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_seconds,
        )
    if settings.llm_provider is LLMProvider.OPENAI_COMPATIBLE:
        key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
        return OpenAICompatibleGenerator(
            base_url=settings.openai_base_url,
            api_key=key,
            model=settings.openai_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_seconds,
        )
    return ExtractiveGenerator()
