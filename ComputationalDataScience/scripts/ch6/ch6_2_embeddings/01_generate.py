"""
STAT 418 · Chapter 6.2 — Embeddings & Feature Extraction
Script 01: Generating embeddings (single, batch, with metadata).

Backs the §6.2 "Generating embeddings" slide. Demonstrates:
  - ai.embed(text)            -> one dense vector (a list of floats),
  - ai.embed([t1, t2, ...])   -> a list of vectors in a single request,
  - ai.embed_complete(texts)  -> an EmbeddingResponse with model / dimension / tokens.

Unlike chat, embeddings are deterministic for a given text + model. The dimension
is a property of the model: llama3.2:latest yields 3072-d vectors here. Always
check len(embedding) when you switch models.

Prereqs: pip install -r ../requirements.txt ; export GENAI_STUDIO_API_KEY=sk-...
Run:     python 01_generate.py
"""
from __future__ import annotations

import os
import sys

from genai_studio import GenAIStudio

EMBED_MODEL = "llama3.2:latest"


def make_client() -> GenAIStudio:
    if not os.environ.get("GENAI_STUDIO_API_KEY"):
        sys.exit("GENAI_STUDIO_API_KEY is not set — see ch6_1_foundations/01_first_chat.py.")
    ai = GenAIStudio()
    ai.select_model(EMBED_MODEL)
    return ai


def single(ai: GenAIStudio) -> None:
    print("--- single embedding ---")
    embedding = ai.embed(
        "Statistical significance measures the probability of observing data "
        "this extreme under the null hypothesis."
    )
    print(f"Type: {type(embedding).__name__}")
    print(f"Dimensions: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")


def batch(ai: GenAIStudio) -> list[str]:
    print("\n--- batch embedding (one request) ---")
    texts = [
        "The mean is sensitive to outliers.",
        "The median is robust to extreme values.",
        "Bootstrap resampling estimates the sampling distribution.",
        "Neural networks learn hierarchical representations.",
        "The recipe calls for two cups of flour.",
    ]
    embeddings = ai.embed(texts)
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Each has {len(embeddings[0])} dimensions")
    return texts


def with_metadata(ai: GenAIStudio, texts: list[str]) -> None:
    print("\n--- embed_complete (metadata) ---")
    response = ai.embed_complete(texts)
    print(f"Model: {response.model}")
    print(f"Dimension: {response.dimension}")
    print(f"Tokens processed: {response.prompt_tokens}")
    print(f"Number of embeddings: {len(response)}")
    print(f"First embedding length: {len(response[0])}")


def main() -> None:
    ai = make_client()
    single(ai)
    texts = batch(ai)
    with_metadata(ai, texts)


if __name__ == "__main__":
    main()
