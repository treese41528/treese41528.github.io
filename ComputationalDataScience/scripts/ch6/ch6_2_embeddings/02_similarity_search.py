"""
STAT 418 · Chapter 6.2 — Embeddings & Feature Extraction
Script 02: Similarity search — cosine similarity, a pairwise similarity matrix,
and nearest-neighbor retrieval (the core idea behind RAG, §6.5).

Demonstrates: manual cosine similarity with numpy, the GenAIStudio.cosine_similarity
static method, the ai.similarity() shortcut (embed + compare in one call), a
pairwise similarity matrix, and ranking a corpus against a query.

Prereqs / run: see 01_generate.py.
"""
from __future__ import annotations

import os
import sys

import numpy as np

from genai_studio import GenAIStudio

EMBED_MODEL = "llama3.2:latest"


def make_client() -> GenAIStudio:
    if not os.environ.get("GENAI_STUDIO_API_KEY"):
        sys.exit("GENAI_STUDIO_API_KEY is not set — see ch6_1_foundations/01_first_chat.py.")
    ai = GenAIStudio()
    ai.select_model(EMBED_MODEL)
    return ai


def cosine_sim(a, b) -> float:
    """Cosine of the angle between two vectors: dot(a, b) / (|a| |b|)."""
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def pairwise_examples(ai: GenAIStudio) -> None:
    print("--- cosine similarity ---")
    e1 = ai.embed("The mean is sensitive to outliers.")
    e2 = ai.embed("The average is affected by extreme values.")
    e3 = ai.embed("The recipe calls for two cups of flour.")
    print(f"mean vs. average: {cosine_sim(e1, e2):.4f}   (manual numpy)")
    print(f"mean vs. recipe:  {cosine_sim(e1, e3):.4f}   (manual numpy)")
    print(f"mean vs. average: {GenAIStudio.cosine_similarity(e1, e2):.4f}   (SDK static method)")
    sim = ai.similarity(
        "The mean is sensitive to outliers.",
        "The average is affected by extreme values.",
    )
    print(f"mean vs. average: {sim:.4f}   (ai.similarity shortcut)")


def similarity_matrix(ai: GenAIStudio) -> None:
    print("\n--- pairwise similarity matrix ---")
    texts = ["machine learning", "deep learning", "neural network",
             "linear regression", "banana split", "ice cream"]
    embeddings = ai.embed(texts)
    n = len(texts)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim[i, j] = GenAIStudio.cosine_similarity(embeddings[i], embeddings[j])
    for label, row in zip(texts, np.round(sim, 2)):
        print(f"  {label:>18}  {row}")


def nearest_neighbor(ai: GenAIStudio) -> None:
    print("\n--- nearest-neighbor search ---")
    corpus = [
        "The bootstrap resamples data with replacement to estimate variability.",
        "Cross-validation partitions data into training and test folds.",
        "Bayesian inference updates prior beliefs with observed data.",
        "Neural networks learn features through gradient descent.",
        "Photosynthesis converts sunlight into chemical energy.",
        "The permutation test shuffles labels to build a null distribution.",
    ]
    corpus_embeddings = ai.embed(corpus)
    query = "How do I estimate uncertainty in my statistic?"
    query_embedding = ai.embed(query)
    sims = [GenAIStudio.cosine_similarity(query_embedding, ce) for ce in corpus_embeddings]
    ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
    print(f"Query: {query}")
    print("Most similar:")
    for idx, s in ranked[:3]:
        print(f"  [{s:.4f}] {corpus[idx]}")


def main() -> None:
    ai = make_client()
    pairwise_examples(ai)
    similarity_matrix(ai)
    nearest_neighbor(ai)


if __name__ == "__main__":
    main()
