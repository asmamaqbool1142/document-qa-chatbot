"""No-LLM fallback: answer by quoting the most relevant retrieved sentences.

This mode needs no model, API key or network. It cannot paraphrase, combine
facts or reason; it simply returns verbatim sentences that share the most
words with the question. Useful for offline demos, tests and as a baseline.
"""

from __future__ import annotations

from app.generation.base import ProviderStatus
from app.generation.text_utils import overlap_score, split_sentences
from app.models.domain import ContextSource, GeneratedAnswer
from app.retrieval.embeddings import tokenize

MAX_SENTENCES = 3
MIN_OVERLAP = 0.34  # at least about a third of the question's content words


class ExtractiveGenerator:
    name = "extractive"
    model = "extractive-sentence-selection"
    is_external = False

    def check(self) -> ProviderStatus:
        return ProviderStatus(True, "Extractive mode (no LLM): answers quote retrieved sentences")

    def generate(self, question: str, sources: list[ContextSource]) -> GeneratedAnswer:
        q_tokens = set(tokenize(question))
        scored: list[tuple[float, int, str, str]] = []
        for order, src in enumerate(sources):
            for sentence in split_sentences(src.text):
                overlap = overlap_score(q_tokens, sentence)
                if overlap >= MIN_OVERLAP:
                    # Combine word overlap with the retriever's ranking signal.
                    score = overlap + 0.25 * src.chunk.score - 0.01 * order
                    scored.append((score, order, sentence, src.label))
        if not scored:
            return GeneratedAnswer("", True, self.name, self.model)

        scored.sort(key=lambda s: s[0], reverse=True)
        picked: list[tuple[float, int, str, str]] = []
        seen: set[str] = set()
        for item in scored:
            key = item[2].lower()
            if key not in seen:
                picked.append(item)
                seen.add(key)
            if len(picked) == MAX_SENTENCES:
                break
        # The response's provider field ("extractive") tells clients no LLM was used.
        text = "\n".join(f"- {sentence} [{label}]" for _, _, sentence, label in picked)
        return GeneratedAnswer(text, False, self.name, self.model)
