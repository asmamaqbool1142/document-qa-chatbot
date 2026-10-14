"""Evaluation dataset schema and loading."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class Evidence(BaseModel):
    file: str
    page: int | None = None
    contains: str = Field(min_length=3)


class EvalCase(BaseModel):
    id: str
    category: str
    question: str
    answerable: bool
    evidence: list[Evidence] = Field(default_factory=list)
    answer_must_contain: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> EvalCase:
        if self.answerable and not self.evidence:
            raise ValueError(f"{self.id}: answerable cases need evidence")
        if not self.answerable and self.evidence:
            raise ValueError(f"{self.id}: unanswerable cases must not list evidence")
        return self


class EvalDataset(BaseModel):
    name: str
    version: int
    description: str = ""
    documents: list[str]
    cases: list[EvalCase]


def load_dataset(path: Path) -> tuple[EvalDataset, str]:
    """Return the dataset and a short SHA-256 of the file for reproducibility."""
    raw = path.read_bytes()
    dataset = EvalDataset.model_validate(json.loads(raw))
    ids = [c.id for c in dataset.cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case ids in dataset")
    return dataset, hashlib.sha256(raw).hexdigest()[:12]
