"""Logging setup.

Logs carry request metadata (method, path, status, timing, IDs) but never
document text, questions, answers or credentials.
"""

from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if getattr(root, "_rag_configured", False):
        root.setLevel(level.upper())
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    root.addHandler(handler)
    root.setLevel(level.upper())
    root._rag_configured = True  # type: ignore[attr-defined]
    # Third-party libraries can be chatty; keep them at WARNING.
    for noisy in ("httpx", "httpx2", "chromadb", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
