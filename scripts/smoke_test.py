"""End-to-end smoke test against a *running* backend.

    python scripts/smoke_test.py                      # API at http://localhost:8000
    python scripts/smoke_test.py --api-url http://localhost:8000 --keep

Uploads the sample documents, asks one answerable and one unanswerable
question, checks citations, then deletes what it uploaded (unless --keep).
Documents that already exist are reused, so it is safe to run repeatedly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"


def check(condition: bool, message: str) -> None:
    print(("  ok   " if condition else "  FAIL ") + message)
    if not condition:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--keep", action="store_true", help="don't delete uploaded samples")
    args = parser.parse_args()
    api = httpx.Client(base_url=args.api_url, timeout=300)

    print(f"Smoke test against {args.api_url}")
    health = api.get("/health").json()
    check(health["embedding"]["ok"] and health["vector_store"]["ok"], "backend healthy")
    print(f"       llm: {health['llm_provider']} - {health['llm']['detail']}")

    uploaded: list[str] = []
    for path in sorted(SAMPLES.glob("*")):
        if path.suffix not in (".pdf", ".txt"):
            continue
        mime = "application/pdf" if path.suffix == ".pdf" else "text/plain"
        resp = api.post("/documents/upload", files={"file": (path.name, path.read_bytes(), mime)})
        if resp.status_code == 409:
            print(f"       {path.name} already indexed")
            continue
        check(resp.status_code == 201, f"upload {path.name} -> {resp.status_code}")
        uploaded.append(resp.json()["document_id"])

    docs = api.get("/documents").json()
    check(docs["total"] >= 3, f"{docs['total']} documents listed")

    if not health["llm"]["ok"]:
        print("  skip chat checks: LLM provider not ready")
    else:
        q = "How often should the reverse osmosis membrane be replaced?"
        body = api.post("/chat", json={"question": q}).json()
        print(f"       Q: {q}\n       A: {body['answer']}")
        check(body["status"] == "answered", "answerable question answered")
        cit = body["citations"][0]
        check(
            cit["filename"] == "aquapure_w300_manual.pdf" and cit["page_number"] == 3,
            f"cited {cit['filename']} page {cit['page_number']}",
        )
        body = api.post("/chat", json={"question": "What is the capital city of Australia?"}).json()
        check(body["status"] == "insufficient_evidence", "off-topic question refused")

    if not args.keep:
        for doc_id in uploaded:
            check(api.delete(f"/documents/{doc_id}").status_code == 200, f"deleted {doc_id}")
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
