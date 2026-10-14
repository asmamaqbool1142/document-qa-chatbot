"""Embedding models behind a single small interface.

``SentenceTransformerEmbedder`` is the default and what you should use for real
retrieval quality. ``HashingEmbedder`` is a dependency-free, deterministic
lexical embedder (feature hashing of words and word pairs). It exists so that
tests and offline environments work without downloading a neural model. It
only matches *words*, not meaning, so expect noticeably weaker retrieval.

The same embedder must be used for indexing and querying; the vector store
records the embedder's ``model_id`` and refuses to mix them.
"""

from __future__ import annotations

import hashlib
import itertools
import logging
import math
import re
from collections import Counter
from typing import Any, Protocol, runtime_checkable

from app.core.config import EmbeddingProvider, Settings
from app.core.errors import ConfigurationError, ProviderUnavailableError

logger = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    @property
    def model_id(self) -> str:
        """Stable identifier stored alongside the vectors."""
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """Local neural embeddings via the ``sentence-transformers`` package."""

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 32) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ConfigurationError(
                "EMBEDDING_PROVIDER=sentence_transformers but the 'sentence-transformers' "
                "package is not installed. Run: pip install -r requirements.txt"
            ) from exc
        try:
            self._model: Any = SentenceTransformer(model_name, device=device)
        except Exception as exc:
            raise ProviderUnavailableError(
                f"Could not load embedding model '{model_name}'. The first run downloads it "
                "from Hugging Face, so check your network connection or pre-download it. "
                f"({type(exc).__name__})"
            ) from exc
        self._model_name = model_name
        self._batch_size = batch_size

    @property
    def model_id(self) -> str:
        return f"sentence_transformers:{self._model_name}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(x) for x in row] for row in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
_STOPWORDS = frozenset(
    (
        "a an the and or but if of to in on at by for with from as is are was were be been "
        "being it its this that these those there their they them he she we you your our i me "
        "my do does did doing have has had having not no so than too very can will would "
        "should could what which who whom whose when where why how all any each few more most "
        "other some such into about over after before between through during above below up "
        "down out off again further then once here just also only own same s t"
    ).split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class HashingEmbedder:
    """Deterministic bag-of-words embedding using the hashing trick.

    Each unigram and bigram is hashed to a bucket with a +/-1 sign; counts are
    log-scaled and the vector is L2-normalised so cosine similarity works.
    """

    def __init__(self, dimensions: int = 1024) -> None:
        self._dims = dimensions

    @property
    def model_id(self) -> str:
        return f"hashing:v1:{self._dims}"

    def _features(self, text: str) -> Counter[str]:
        tokens = tokenize(text)
        feats: Counter[str] = Counter(tokens)
        feats.update(f"{a}_{b}" for a, b in itertools.pairwise(tokens))
        return feats

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dims
        for feat, count in self._features(text).items():
            digest = hashlib.blake2b(feat.encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self._dims
            sign = 1.0 if digest[4] & 1 else -1.0
            weight = 1.0 + math.log(count)
            if "_" in feat:
                weight *= 0.5  # bigrams help with phrases but shouldn't dominate
            vec[bucket] += sign * weight
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            return vec
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider is EmbeddingProvider.HASHING:
        return HashingEmbedder(settings.hashing_dimensions)
    logger.info(
        "Loading embedding model %s on %s", settings.embedding_model, settings.embedding_device
    )
    return SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_device)
