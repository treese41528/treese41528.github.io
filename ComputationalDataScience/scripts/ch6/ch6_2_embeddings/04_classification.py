"""
STAT 418 · Chapter 6.2 — Embeddings & Feature Extraction
Script 04: Classification on embeddings — turn text classification into a
standard supervised-learning problem.

Once texts are embedded, any classifier from the earlier chapters applies. Here
we embed REAL labeled product reviews (the shared Chapter 6 corpus — 9,000 real
Amazon reviews, see _data/DATA_CARD.md), evaluate a logistic-regression sentiment
classifier with cross-validation (Chapter 4), then predict on held-out reviews and
read off the calibrated probabilities (Chapter 6.8).

We use a class-balanced sample of positive vs. negative reviews (gold sentiment
derived from the real star rating: 4-5 = positive, 1-2 = negative). Embeddings
are deterministic per text + model, so the cross-validated accuracy reproduces
exactly on re-run with the same model.

Prereqs / run: see 01_generate.py. (Also needs scikit-learn.)
"""
from __future__ import annotations

import os
import sys

import numpy as np

from genai_studio import GenAIStudio

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_data"))
from loader import load_reviews, balanced_sample  # noqa: E402

EMBED_MODEL = "llama3.2:latest"
PER_CLASS = 200          # 200 positive + 200 negative = 400 labeled reviews


def make_client() -> GenAIStudio:
    if not os.environ.get("GENAI_STUDIO_API_KEY"):
        sys.exit("GENAI_STUDIO_API_KEY is not set — see ch6_1_foundations/01_first_chat.py.")
    ai = GenAIStudio()
    ai.select_model(EMBED_MODEL)
    return ai


def embed_all(ai: GenAIStudio, texts: list[str], batch: int = 32) -> np.ndarray:
    """Embed many texts in batches (one API call per batch)."""
    vecs: list[list[float]] = []
    for i in range(0, len(texts), batch):
        vecs.extend(ai.embed(texts[i:i + batch]))
        print(f"  embedded {min(i + batch, len(texts))}/{len(texts)}", end="\r")
    print()
    return np.array(vecs)


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    ai = make_client()
    reviews = load_reviews()

    # Binary sentiment: positive vs. negative (drop the 3-star neutral middle for a
    # clean two-class problem). Class-balanced and reproducible (seed=42).
    posneg = [r for r in reviews if r["sentiment"] in ("positive", "negative")]
    train = balanced_sample(posneg, per_class=PER_CLASS, classes=("positive", "negative"))
    train_ids = {r["id"] for r in train}

    texts_train = [r["text"] for r in train]
    y_train = np.array([1 if r["sentiment"] == "positive" else 0 for r in train])

    print(f"Embedding {len(texts_train)} real reviews with {EMBED_MODEL} ...")
    X_train = embed_all(ai, texts_train)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy")
    print(f"Cross-validated accuracy (n={len(y_train)}): {scores.mean():.3f} +/- {scores.std():.3f}")

    clf.fit(X_train, y_train)

    # Held-out real reviews: a clear positive (5★), a clear negative (1★), and an
    # ambiguous 3-star — to show a confident vs. an uncertain calibrated probability.
    def pick(star):
        for r in reviews:
            if r["rating"] == star and r["id"] not in train_ids:
                return r
        return None
    held = [pick(5), pick(1), pick(3)]
    new_texts = [r["text"] for r in held]
    X_new = embed_all(ai, new_texts)
    predictions = clf.predict(X_new)
    probas = clf.predict_proba(X_new)

    print()
    for r, pred, prob in zip(held, predictions, probas):
        label = "positive" if pred == 1 else "negative"
        print(f"  [{label} (p={prob[pred]:.3f})] ({r['rating']}★) {r['text'][:80]}")


if __name__ == "__main__":
    main()
