"""
STAT 418 · Chapter 6.5 — Retrieval-Augmented Generation
Script 02: Manual RAG from scratch — chunk, embed, retrieve, generate.

Builds the RAG pipeline by hand so the mechanics are visible: chunk documents
(Section 6.3), embed the chunks (Section 6.2), embed the query, retrieve the
top-k by cosine similarity, and generate a grounded answer. The SimpleRAG class
(in rag_core.py) packages these steps; we also show how chunk size shifts which
sources are retrieved.

Uses llama3.2:latest throughout: retrieval needs an embedding-capable model, and
gemma3 (the book's chat model) does not expose an embedding endpoint.

Prereqs / run: see ../ch6_1_foundations/01_first_chat.py. (numpy; rag_core.py sibling file.)
"""
from __future__ import annotations

import os
import sys

import numpy as np

from genai_studio import GenAIStudio
from rag_core import DOCUMENTS, SimpleRAG, build_rag_prompt, chunk_document, retrieve

MODEL = "llama3.2:latest"


def make_client():
    if not os.environ.get("GENAI_STUDIO_API_KEY"):
        sys.exit("GENAI_STUDIO_API_KEY is not set - see ../ch6_1_foundations/01_first_chat.py.")
    ai = GenAIStudio()
    ai.select_model(MODEL)
    return ai


def main():
    ai = make_client()

    # --- The pipeline, one step at a time ---
    all_chunks, chunk_sources = [], []
    for name, content in DOCUMENTS.items():
        for ch in chunk_document(content, chunk_size=50, overlap=10):
            all_chunks.append(ch)
            chunk_sources.append(name)
    chunk_matrix = np.array(ai.embed(all_chunks))
    print(f"Indexed {len(all_chunks)} chunks, matrix shape {chunk_matrix.shape}")

    query = "How do I construct a confidence interval without assumptions?"
    retrieved = retrieve(query, chunk_matrix, all_chunks, chunk_sources, ai, top_k=3)
    print(f"\nQuery: {query}\nRetrieved top-3:")
    for r in retrieved:
        print(f"  [{r['similarity']:.4f}] [{r['source']}] {r['chunk'][:62]}...")

    try:
        answer = ai.chat(build_rag_prompt(query, retrieved))
        print(f"\nGrounded answer:\n  {answer[:300]}")
    except Exception as exc:
        print(f"\nGrounded answer: [call dropped: {exc}]")

    # --- Same thing via the SimpleRAG class ---
    rag = SimpleRAG(ai)
    for name, content in DOCUMENTS.items():
        rag.add_document(content, source=name)
    rag.build_index()
    try:
        result = rag.query("What is MCMC used for?")
        print(f"\nSimpleRAG answer:\n  {result['answer'][:220]}")
        print(f"  sources used: {[r['source'] for r in result['sources']]}")
    except Exception as exc:
        print(f"\nSimpleRAG answer: [call dropped: {exc}]")

    # --- Chunk size shifts retrieval focus ---
    print("\nChunk-size effect (query: how the bootstrap builds CIs):")
    q = "How does the bootstrap construct confidence intervals?"
    for cs in [30, 75, 150]:
        rt = SimpleRAG(ai)
        for name, content in DOCUMENTS.items():
            rt.add_document(content, source=name, chunk_size=cs)
        rt.build_index()
        hits = rt.retrieve_chunks(q, top_k=3)  # retrieval only - no generation needed
        sims = [r["similarity"] for r in hits]
        print(f"  size={cs:3d}: {len(rt.chunks)} chunks, top sim {max(sims):.4f}, "
              f"sources={[r['source'] for r in hits]}")


if __name__ == "__main__":
    main()
