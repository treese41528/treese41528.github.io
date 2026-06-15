"""
STAT 418 · Chapter 6.5 — Retrieval-Augmented Generation
Script 03: Evaluating RAG — retrieval metrics and answer faithfulness.

A RAG system has two things to evaluate: retrieval (did we fetch the right
chunks?) and generation (is the answer grounded in them?). We measure
Precision@k / Recall@k / F1 over labeled queries, then use a judge model to
verify a generated answer is faithful to its retrieved context.

Uses llama3.2:latest (retrieval needs an embedding-capable model; gemma3 has no
embedding endpoint).

Prereqs / run: see ../ch6_1_foundations/01_first_chat.py. (numpy; rag_core.py sibling file.)
"""
from __future__ import annotations

import os
import sys

import numpy as np

from genai_studio import GenAIStudio
from rag_core import DOCUMENTS, SimpleRAG

MODEL = "llama3.2:latest"


def evaluate_retrieval(queries, expected_sources, rag_system, top_k=3) -> dict:
    """Precision@k, Recall@k, and F1 over labeled queries."""
    precisions, recalls = [], []
    for query, expected in zip(queries, expected_sources):
        sources = [r["source"] for r in rag_system.retrieve_chunks(query, top_k=top_k)]
        hits = sum(1 for s in sources if s in expected)
        precisions.append(hits / len(sources))
        recalls.append(hits / len(expected) if expected else 0)
    mp, mr = float(np.mean(precisions)), float(np.mean(recalls))
    return {"mean_precision": mp, "mean_recall": mr,
            "mean_f1": 2 * mp * mr / (mp + mr + 1e-10)}


def check_faithfulness(answer: str, context: str, ai) -> str:
    """Ask a judge model whether every claim in the answer is supported."""
    prompt = f"""You are a fact-checking assistant. Given a context and an answer,
determine whether every claim in the answer is supported by the context.

### Context ###
{context}
### End Context ###

### Answer to Check ###
{answer}
### End Answer ###

Respond with ONLY one of:
- FAITHFUL: All claims are supported by the context
- PARTIALLY_FAITHFUL: Some claims are supported, some are not
- UNFAITHFUL: The answer contains claims not in the context

Verdict:"""
    try:
        raw = ai.chat(prompt).strip().upper()
    except Exception as exc:
        return f"[judge call dropped: {exc}]"
    # Some judge models add a justification; pull out the verdict keyword.
    # (UNFAITHFUL / PARTIALLY_FAITHFUL both contain "FAITHFUL", so test them first.)
    for verdict in ("PARTIALLY_FAITHFUL", "UNFAITHFUL", "FAITHFUL"):
        if verdict in raw:
            return verdict
    return raw[:40]


def make_client():
    if not os.environ.get("GENAI_STUDIO_API_KEY"):
        sys.exit("GENAI_STUDIO_API_KEY is not set - see ../ch6_1_foundations/01_first_chat.py.")
    ai = GenAIStudio()
    ai.select_model(MODEL)
    return ai


def main():
    ai = make_client()
    rag = SimpleRAG(ai)
    for name, content in DOCUMENTS.items():
        # Smaller chunks -> more chunks than top_k, so Precision@k is meaningful
        # (with one chunk per doc, top-3 trivially returns every source).
        rag.add_document(content, source=name, chunk_size=40)
    rag.build_index()

    # --- Retrieval metrics: each query should hit one known source ---
    queries = ["How does bootstrap resampling work?",
               "What is a prior distribution?",
               "How many folds in cross-validation?"]
    expected = [["bootstrap.txt"], ["bayesian.txt"], ["cross_validation.txt"]]
    m = evaluate_retrieval(queries, expected, rag)
    print(f"Precision@3: {m['mean_precision']:.3f}")
    print(f"Recall@3:    {m['mean_recall']:.3f}")
    print(f"F1@3:        {m['mean_f1']:.3f}")
    print("  (Recall@k: was the right source retrieved?  Precision@k: how focused was")
    print("   retrieval?  Related topics embed near each other (6.2) - lifting recall")
    print("   while depressing precision.)")

    # --- Faithfulness of a generated answer ---
    try:
        result = rag.query("What is the bootstrap?")
        context = "\n".join(r["chunk"] for r in result["sources"])
        print(f"\nAnswer: {result['answer'][:180]}...")
        print(f"Faithfulness verdict: {check_faithfulness(result['answer'], context, ai)}")
        print("  (judge-dependent: a strict judge flags any detail not stated verbatim")
        print("   in the retrieved context.)")
    except Exception as exc:
        print(f"\n[generation/faithfulness demo dropped: {exc}]")


if __name__ == "__main__":
    main()
