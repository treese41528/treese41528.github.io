"""
STAT 418 · Chapter 6.3 — Text Preprocessing for LLM Pipelines
Script 02: Chunking strategies — splitting long documents to fit the context window.

Three strategies, from crude to structure-aware: fixed-size, fixed-size with
overlap, and semantic (split at paragraph boundaries). The final step embeds the
chunks from each strategy and compares their average similarity to the full
document — a quantitative look at which strategy best preserves meaning
(Exercise 6.3.2).

The chunking itself needs no API; the embedding comparison runs only if
GENAI_STUDIO_API_KEY is set.

Prereqs / run: see ../ch6_1_foundations/01_first_chat.py. (numpy for the comparison.)
"""
from __future__ import annotations

import os

import numpy as np

from genai_studio import GenAIStudio

EMBED_MODEL = "llama3.2:latest"


def chunk_fixed_size(text: str, chunk_size: int = 500) -> list:
    """Split into chunks of ~chunk_size words. Simple, but may split mid-sentence."""
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


def chunk_with_overlap(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """Split into overlapping chunks so information at boundaries is not lost."""
    words = text.split()
    chunks, step = [], chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks


def chunk_semantic(text: str, max_chunk_size: int = 500) -> list:
    """Split at paragraph boundaries, merging small paragraphs up to max_chunk_size."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current, size = [], [], 0
    for para in paragraphs:
        para_size = len(para.split())
        if size + para_size > max_chunk_size and current:
            chunks.append("\n\n".join(current))
            current, size = [para], para_size
        else:
            current.append(para)
            size += para_size
    if current:
        chunks.append("\n\n".join(current))
    return chunks


STRUCTURED_TEXT = """
Introduction to Bootstrap Methods

The bootstrap is a resampling method introduced by Bradley Efron in 1979.
It estimates the sampling distribution of a statistic by repeatedly
resampling from the observed data with replacement.

How the Bootstrap Works

Given a dataset of n observations, we draw n samples with replacement
to create a bootstrap sample. We compute the statistic of interest
on this sample. Repeating this B times gives B bootstrap estimates.

Confidence Intervals

The percentile method takes the alpha/2 and 1-alpha/2 quantiles of
the bootstrap distribution as confidence interval endpoints. This is
simple but can be improved with bias-corrected methods.
""".strip()


def main() -> None:
    # --- Fixed-size vs. overlap on a synthetic document ---
    long_text = " ".join(f"Sentence {i} discusses topic {i % 5}." for i in range(200))
    fixed = chunk_fixed_size(long_text, chunk_size=50)
    overlap = chunk_with_overlap(long_text, chunk_size=50, overlap=10)
    print(f"Document: {len(long_text.split())} words")
    print(f"  fixed-size (50w):   {len(fixed)} chunks")
    print(f"  with overlap (10w): {len(overlap)} chunks "
          f"({(len(overlap) / len(fixed) - 1) * 100:.0f}% more, for boundary coverage)")

    # --- Semantic chunking preserves document structure ---
    semantic = chunk_semantic(STRUCTURED_TEXT, max_chunk_size=40)
    print(f"\nSemantic chunking -> {len(semantic)} structure-preserving chunks:")
    for i, chunk in enumerate(semantic):
        print(f"  Chunk {i} ({len(chunk.split())} words): {chunk.splitlines()[0]}")

    # --- Which strategy best preserves the document's meaning? (Exercise 6.3.2) ---
    if not os.environ.get("GENAI_STUDIO_API_KEY"):
        print("\n[embedding comparison skipped - set GENAI_STUDIO_API_KEY to run it]")
        return
    ai = GenAIStudio()
    ai.select_model(EMBED_MODEL)
    doc_emb = ai.embed(STRUCTURED_TEXT)
    print(f"\nAvg chunk-to-document cosine similarity by strategy ({EMBED_MODEL}):")
    for name, chunks in [("fixed", chunk_fixed_size(STRUCTURED_TEXT, 40)),
                         ("overlap", chunk_with_overlap(STRUCTURED_TEXT, 40, 8)),
                         ("semantic", semantic)]:
        embs = ai.embed(chunks)
        sims = [GenAIStudio.cosine_similarity(doc_emb, e) for e in embs]
        print(f"  {name:8s}: {len(chunks)} chunks, avg similarity {np.mean(sims):.4f}")


if __name__ == "__main__":
    main()
