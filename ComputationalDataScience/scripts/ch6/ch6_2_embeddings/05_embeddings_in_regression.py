"""
STAT 418 · Chapter 6.2 — Embeddings & Feature Extraction
Script 05: Embeddings as covariates in a regression — plus bootstrap CIs.

The deepest tie to the earlier course: reduce the 3072-d embedding to a few
principal components, use them (with structured features like review length) as
covariates in a linear model predicting the REAL star rating, then quantify
coefficient uncertainty with the Chapter 4 bootstrap.

Data: the shared Chapter 6 corpus (9,000 real Amazon reviews — see
_data/DATA_CARD.md). We take a sample balanced across the 1-5 star ratings so the
outcome has real spread, with n >> p (default n = 500, p = 6) — so R^2 is an
honest fit, not the inflated value a tiny toy sample produces.

Prereqs / run: see 01_generate.py. (Also needs scikit-learn.)
"""
from __future__ import annotations

import os
import sys

import numpy as np

from genai_studio import GenAIStudio

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_data"))
from loader import load_reviews, stratified_by_rating  # noqa: E402

EMBED_MODEL = "llama3.2:latest"
PER_RATING = 100          # 100 reviews per star (1-5) -> n = 500


def make_client() -> GenAIStudio:
    if not os.environ.get("GENAI_STUDIO_API_KEY"):
        sys.exit("GENAI_STUDIO_API_KEY is not set — see ch6_1_foundations/01_first_chat.py.")
    ai = GenAIStudio()
    ai.select_model(EMBED_MODEL)
    return ai


def embed_all(ai: GenAIStudio, texts: list[str], batch: int = 32) -> np.ndarray:
    vecs: list[list[float]] = []
    for i in range(0, len(texts), batch):
        vecs.extend(ai.embed(texts[i:i + batch]))
        print(f"  embedded {min(i + batch, len(texts))}/{len(texts)}", end="\r")
    print()
    return np.array(vecs)


def main() -> None:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression

    ai = make_client()
    reviews = load_reviews()
    sample = stratified_by_rating(reviews, per_rating=PER_RATING)
    texts = [r["text"] for r in sample]
    RATINGS = np.array([r["rating"] for r in sample])

    # Embed reviews, reduce to 5 PCs, add a structured feature (review length).
    review_lengths = np.array([len(t.split()) for t in texts])
    print(f"Embedding {len(texts)} real reviews with {EMBED_MODEL} ...")
    emb_matrix = embed_all(ai, texts)
    pca = PCA(n_components=5, random_state=42)
    emb_reduced = pca.fit_transform(emb_matrix)

    # Standardize features so coefficients are comparable (effect per 1 SD).
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(np.column_stack([emb_reduced, review_lengths]))
    feature_names = [f"PC{i + 1}" for i in range(5)] + ["review_length"]

    model = LinearRegression()
    model.fit(X, RATINGS)
    print(f"Predicting star rating from embedding PCs (n={len(RATINGS)}, p={X.shape[1]}; standardized betas):")
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
