"""
STAT 418 · Chapter 6.2 — Embeddings & Feature Extraction
Script 05: Embeddings as covariates in a regression — plus bootstrap CIs.

The deepest tie to the earlier course: reduce the 3072-d embedding to a few
principal components, use them (with structured features like review length) as
covariates in a linear model predicting the star rating, then quantify
coefficient uncertainty with the Chapter 4 bootstrap.

NB: this is illustrative — n = 10 documents with 6 predictors leaves only 3
residual d.f., so R^2 is inflated. A real analysis needs n >> p.

Prereqs / run: see 01_generate.py. (Also needs scikit-learn.)
"""
from __future__ import annotations

import os
import sys

import numpy as np

from genai_studio import GenAIStudio

EMBED_MODEL = "llama3.2:latest"

REVIEWS = [
    "Absolutely love this, perfect in every way.",
    "Good quality but shipping was slow.",
    "Decent product for the price.",
    "Not what I expected, somewhat disappointed.",
    "Terrible, complete waste of money.",
    "Great value, exceeded expectations.",
    "It works but feels cheaply made.",
    "Outstanding craftsmanship and design.",
    "Mediocre at best, won't buy again.",
    "Fantastic, my favorite purchase this year.",
]
RATINGS = np.array([5, 4, 3, 2, 1, 5, 3, 5, 2, 5])


def make_client() -> GenAIStudio:
    if not os.environ.get("GENAI_STUDIO_API_KEY"):
        sys.exit("GENAI_STUDIO_API_KEY is not set — see ch6_1_foundations/01_first_chat.py.")
    ai = GenAIStudio()
    ai.select_model(EMBED_MODEL)
    return ai


def main() -> None:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression

    ai = make_client()

    # Embed reviews, reduce to 5 PCs, add a structured feature (review length).
    review_lengths = np.array([len(r.split()) for r in REVIEWS])
    emb_matrix = np.array(ai.embed(REVIEWS))
    pca = PCA(n_components=5, random_state=42)
    emb_reduced = pca.fit_transform(emb_matrix)

    X = np.column_stack([emb_reduced, review_lengths])
    feature_names = [f"PC{i + 1}" for i in range(5)] + ["review_length"]

    model = LinearRegression()
    model.fit(X, RATINGS)
    print("Coefficients:")
    for name, coef in zip(feature_names, model.coef_):
        print(f"  {name:>15}: {coef:+.4f}")
    print(f"  {'intercept':>15}: {model.intercept_:+.4f}")
    print(f"\n  R^2: {model.score(X, RATINGS):.4f}")

    # --- Bootstrap CIs on the coefficients (Chapter 4) ---
    rng = np.random.default_rng(42)  # seeded for reproducibility
    n_bootstrap = 1000
    n = len(RATINGS)
    coef_bootstrap = np.zeros((n_bootstrap, X.shape[1]))
    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        model_boot = LinearRegression().fit(X[idx], RATINGS[idx])
        coef_bootstrap[b] = model_boot.coef_

    print("\n95% Bootstrap Confidence Intervals:")
    for i, name in enumerate(feature_names):
        ci_low = np.percentile(coef_bootstrap[:, i], 2.5)
        ci_high = np.percentile(coef_bootstrap[:, i], 97.5)
        point = np.mean(coef_bootstrap[:, i])
        sig = "***" if ci_low > 0 or ci_high < 0 else ""
        print(f"  {name:>15}: {point:+.4f}  [{ci_low:+.4f}, {ci_high:+.4f}] {sig}")


if __name__ == "__main__":
    main()
