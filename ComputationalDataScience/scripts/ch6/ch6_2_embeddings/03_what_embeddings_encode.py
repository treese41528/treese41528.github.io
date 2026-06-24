"""
STAT 418 · Chapter 6.2 — Embeddings & Feature Extraction
Script 03: What does an embedding encode?

An embedding is not a black box. Run PCA on the 3072-d vectors, then correlate
each principal component with metadata we ALREADY know — review length, sentiment
(derived from the star rating), and product category — to read off what each
direction stores.

The recurring lesson: the highest-variance axes capture verbosity, not the thing
you care about, so a naive PC1×PC2 scatter just shows length while sentiment and
category look like noise. The signal isn't gone, it's buried — project it the
right way (a supervised LDA for category, a logistic model for sentiment) and it
reappears. That is exactly why we feed embeddings to supervised models, not 2-D
pictures (the next scripts, 04 and 05).

Data: the shared Chapter 6 corpus (9,000 real Amazon reviews, 6 modern
categories balanced across 1-5 stars — see _data/DATA_CARD.md).

Numbers depend on the deployed model version, so yours will differ from the
slides — but the *structure* holds: length dominates the leading PCs, sentiment
and category live on later directions, and a supervised model reads both off the
full vector far better than any 2-D plot.

Prereqs / run: see 01_generate.py. (Also needs scikit-learn.)
"""
from __future__ import annotations

import os
import sys

import numpy as np

from genai_studio import GenAIStudio

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_data"))
from loader import load_reviews  # noqa: E402

EMBED_MODEL = "llama3.2:latest"
PER_CATEGORY = 80          # 80 reviews × 6 categories -> n = 480
N_COMPONENTS = 10
SEED = 42

SENT_SCORE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}


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


def sample_by_category(reviews: list[dict], per_category: int, seed: int = SEED):
    """Reproducible sample with equal reviews per product category (each category
    spans the 1-5 stars, so sentiment varies within every category)."""
    import random
    rng = random.Random(seed)
    cats = sorted({r["category"] for r in reviews})
    out: list[dict] = []
    for c in cats:
        pool = [r for r in reviews if r["category"] == c]
        out += rng.sample(pool, min(per_category, len(pool)))
    rng.shuffle(out)
    return out, cats


def correlation_ratio(groups: np.ndarray, values: np.ndarray) -> float:
    """η — strength of association between a categorical grouping and a continuous
    value: sqrt(between-group SS / total SS). 0 = no association, 1 = perfect."""
    grand = values.mean()
    ss_between = sum(len(values[groups == g]) * (values[groups == g].mean() - grand) ** 2
                     for g in np.unique(groups))
    ss_total = ((values - grand) ** 2).sum()
    return float(np.sqrt(ss_between / ss_total)) if ss_total > 0 else 0.0


def main() -> None:
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    ai = make_client()
    reviews = load_reviews()
    sample, cats = sample_by_category(reviews, PER_CATEGORY)
    texts = [r["text"] for r in sample]
    category = np.array([r["category"] for r in sample])
    sentiment = np.array([r["sentiment"] for r in sample])
    length = np.array([len(t.split()) for t in texts], dtype=float)
    sent_score = np.array([SENT_SCORE[s] for s in sentiment], dtype=float)

    print(f"Embedding {len(texts)} real reviews ({len(cats)} categories) with {EMBED_MODEL} ...")
    emb = embed_all(ai, texts)
    pca = PCA(n_components=N_COMPONENTS, random_state=SEED)
    scores = pca.fit_transform(emb)

    # --- Decode each PC: correlate it with length, sentiment, and category ---
    print(f"\nWhat each of the first {N_COMPONENTS} PCs encodes")
    print("  (|Pearson r| with length & signed sentiment; η with category):")
    print(f"  {'PC':>4} {'var%':>6} {'|r| length':>11} {'|r| sent':>9} {'η category':>11}   strongest")
    for i in range(N_COMPONENTS):
        pc = scores[:, i]
        r_len = abs(np.corrcoef(pc, length)[0, 1])
        r_sent = abs(np.corrcoef(pc, sent_score)[0, 1])
        eta_cat = correlation_ratio(category, pc)
        var = pca.explained_variance_ratio_[i] * 100
        tag = max((("length", r_len), ("sentiment", r_sent), ("category", eta_cat)),
                  key=lambda t: t[1])[0]
        print(f"  PC{i + 1:>2} {var:>5.1f}% {r_len:>11.2f} {r_sent:>9.2f} {eta_cat:>11.2f}   <- {tag}")
    print("  Typically: PC1-2 = length (the dominant variance), PC3-5 = sentiment, PC7 = category.")

    # --- Buried, not absent: a SUPERVISED projection recovers the concept ---
    print("\nThe variance ranking hides the signal — a supervised model reads it off the full vector:")
    lda_acc = cross_val_score(
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        emb, category, cv=5).mean()
    print(f"  category  : LDA 5-fold accuracy {lda_acc:.2f}   (chance {1 / len(cats):.2f})")
    mask = sentiment != "neutral"                       # binary pos-vs-neg is the clean signal
    bin_acc = cross_val_score(
        LogisticRegression(max_iter=1000),
        emb[mask], sentiment[mask], cv=5).mean()
    print(f"  sentiment : logistic pos-vs-neg accuracy {bin_acc:.2f}   (chance 0.50)")

    print("\nTakeaway: one 3072-d vector blends length + sentiment + category along different "
          "directions. The dominant variance is length, so the semantic signal is real but "
          "BURIED — feed embeddings to supervised models, not 2-D pictures (see scripts 04-05).")


if __name__ == "__main__":
    main()
