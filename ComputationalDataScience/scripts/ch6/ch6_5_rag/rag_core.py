"""
STAT 418 · Chapter 6.5 — Retrieval-Augmented Generation
Shared RAG building blocks: chunking, retrieval, prompt construction, and a
minimal SimpleRAG pipeline, plus a small sample corpus. Imported by
02_manual_rag.py and 03_rag_evaluation.py (run those; this module is a library).
"""
from __future__ import annotations

import numpy as np

from genai_studio import GenAIStudio


def chunk_document(text: str, chunk_size: int = 200, overlap: int = 30) -> list:
    """Chunk a document into overlapping word segments (see Section 6.3)."""
    words = text.split()
    chunks, step = [], chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks


def retrieve(query, chunk_matrix, all_chunks, chunk_sources, ai, top_k=3) -> list:
    """Retrieve the top-k chunks most similar to the query embedding."""
    query_embedding = ai.embed(query)
    sims = [GenAIStudio.cosine_similarity(query_embedding, c) for c in chunk_matrix]
    ranked = np.argsort(sims)[::-1][:top_k]
    return [{"chunk": all_chunks[i], "source": chunk_sources[i],
             "similarity": float(sims[i])} for i in ranked]


def build_rag_prompt(query: str, retrieved_chunks: list) -> str:
    """Construct a prompt that grounds the answer in the retrieved context."""
    context = "\n\n".join(f"[Source: {r['source']}]\n{r['chunk']}" for r in retrieved_chunks)
    return f"""Answer the following question based ONLY on the provided context.
If the context does not contain the answer, say "I cannot answer this based on
the provided documents."

### Context ###
{context}
### End Context ###

Question: {query}

Answer:"""


class SimpleRAG:
    """Minimal RAG pipeline built on GenAI Studio embeddings."""

    def __init__(self, ai):
        self.ai = ai
        self.chunks = []
        self.sources = []
        self.embeddings = None

    def add_document(self, text, source="unknown", chunk_size=100, overlap=15):
        new_chunks = chunk_document(text, chunk_size, overlap)
        self.chunks.extend(new_chunks)
        self.sources.extend([source] * len(new_chunks))

    def build_index(self):
        self.embeddings = np.array(self.ai.embed(self.chunks))
        return self.embeddings.shape

    def retrieve_chunks(self, question, top_k=3):
        """Retrieve the top-k chunks only (no generation) - for retrieval-quality eval."""
        return retrieve(question, self.embeddings, self.chunks, self.sources, self.ai, top_k)

    def query(self, question, top_k=3):
        retrieved = retrieve(question, self.embeddings, self.chunks,
                             self.sources, self.ai, top_k)
        answer = self.ai.chat(build_rag_prompt(question, retrieved))
        return {"answer": answer, "sources": retrieved}


# Sample corpus reused by the demo and evaluation scripts.
DOCUMENTS = {
    "bootstrap.txt": (
        "The bootstrap is a resampling method introduced by Bradley Efron in 1979. "
        "It estimates the sampling distribution of a statistic by repeatedly drawing "
        "samples with replacement from the observed data. Given n observations, a "
        "bootstrap sample draws n values with replacement, computes the statistic, and "
        "repeats B times. The resulting B estimates approximate the sampling "
        "distribution. Bootstrap confidence intervals can be constructed using the "
        "percentile method, the basic method, or the bias-corrected and accelerated "
        "(BCa) method. The bootstrap is distribution-free; it makes no assumptions "
        "about the population distribution."
    ),
    "bayesian.txt": (
        "Bayesian inference treats parameters as random variables with prior "
        "distributions that are updated via Bayes' theorem after observing data. The "
        "prior encodes what we believe before seeing data; the likelihood describes how "
        "probable the data is given the parameters; and the posterior combines both. "
        "Markov Chain Monte Carlo (MCMC) methods, particularly Metropolis-Hastings and "
        "Hamiltonian Monte Carlo (HMC), are used to sample from posterior distributions "
        "when analytical solutions are unavailable. Bayesian credible intervals have a "
        "direct probability interpretation, unlike frequentist confidence intervals."
    ),
    "cross_validation.txt": (
        "Cross-validation estimates out-of-sample prediction error by partitioning data "
        "into training and validation folds. In k-fold cross-validation, the data is "
        "split into k equally-sized folds. The model is trained on k-1 folds and "
        "evaluated on the held-out fold, rotating through all k folds. The average error "
        "across folds estimates generalization performance. Leave-one-out "
        "cross-validation (LOOCV) is the special case where k equals n. Stratified "
        "k-fold ensures class proportions are maintained in each fold."
    ),
}
