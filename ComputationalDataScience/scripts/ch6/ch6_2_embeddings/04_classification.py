"""
STAT 418 · Chapter 6.2 — Embeddings & Feature Extraction
Script 04: Classification on embeddings — turn text classification into a
standard supervised-learning problem.

Once texts are embedded, any classifier from the earlier chapters applies. Here
we embed labeled product reviews, evaluate a logistic-regression sentiment
classifier with cross-validation (Chapter 4), then predict on new reviews and
read off the calibrated probabilities (Chapter 6.7).

Prereqs / run: see 01_generate.py. (Also needs scikit-learn.)
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


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    ai = make_client()

    texts_train = [
        "Great product, exactly what I needed!", "Terrible quality, broke after a week.",
        "Solid purchase, would buy again.", "Waste of money, very disappointed.",
        "Love it, exceeded my expectations!", "Awful experience, avoid this.",
        "Perfect gift, my friend was thrilled.", "Poor craftsmanship, returned it.",
        "Excellent value for the price.", "Defective on arrival, total junk.",
        "Best purchase I've made this year.", "Completely useless, don't bother.",
        "Highly recommend to everyone.", "Regret buying this, cheap materials.",
        "Amazing quality, fast shipping too!", "Fell apart immediately, so frustrating.",
    ]
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

    X_train = np.array(ai.embed(texts_train))
    y_train = np.array(labels)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    scores = cross_val_score(clf, X_train, y_train, cv=4, scoring="accuracy")
    print(f"Cross-validated accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")

    clf.fit(X_train, y_train)
    new_texts = [
        "Wonderful product, highly satisfied!",
        "Broke on the first use, terrible.",
        "It's okay, nothing special.",
    ]
    X_new = np.array(ai.embed(new_texts))
    predictions = clf.predict(X_new)
    probas = clf.predict_proba(X_new)

    print()
    for text, pred, prob in zip(new_texts, predictions, probas):
        label = "positive" if pred == 1 else "negative"
        print(f"  [{label} (p={prob[pred]:.3f})] {text}")


if __name__ == "__main__":
    main()
