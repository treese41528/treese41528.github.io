"""
STAT 418 · Chapter 6.2 — Embeddings & Feature Extraction
Script 06: The geometry of "concept arithmetic" — does king - man + woman = queen?

This is the demo behind the §6.2 caveat box. It contrasts three things:

  1. STATIC word vectors (Word2Vec/GloVe via gensim) — the classic setting where
     "king - man + woman = queen" works and the gender offsets are strongly
     parallel. Our positive control. It also shows the well-known catch
     (Drozd et al. 2016; Nissim et al. 2020): the result depends on *excluding*
     the three input words; without that, the nearest vector is often an input.
  2. CONTEXTUAL embeddings (GenAI Studio, e.g. llama3.2) — where the same additive
     arithmetic is far noisier and the gender offsets are only weakly parallel.
  3. THREE ways to model the male->female relation — an additive offset, an
     orthogonal rotation (Procrustes), and a learned linear map — scored
     leave-one-out, on two models, showing there is no single "right" operator.

Takeaway: relationships ARE encoded geometrically, but `+/-` assumes a clean
parallel *translation* that contextual embeddings only approximate. See §6.2
(the caveat box) and the §6.9 Further Reading interpretability entries for *why*
(superposition). Refs: Mikolov et al. 2013; Drozd et al. 2016; Nissim et al.
2020; Allen & Hospedales 2019; Ethayarajh 2019.

Prereqs / run: see 01_generate.py for the GenAI Studio setup. This script ALSO
needs gensim for the static-vector control (the one place we step outside the
genai_studio SDK):
    pip install gensim
The first run downloads a ~128 MB GloVe model, then caches it.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from numpy.linalg import lstsq, norm, svd

from genai_studio import GenAIStudio

EMBED_MODEL = "llama3.2:latest"                       # primary contextual model
COMPARE_MODELS = ["llama3.2:latest", "phi4:latest"]   # to show model-dependence

# Classic gender analogy pairs (male -> female).
PAIRS = [
    ("man", "woman"), ("king", "queen"), ("boy", "girl"),
    ("prince", "princess"), ("actor", "actress"), ("uncle", "aunt"),
    ("brother", "sister"), ("father", "mother"), ("nephew", "niece"),
]


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (norm(a) * norm(b)))


def offset_parallelism(males: np.ndarray, females: np.ndarray) -> float:
    """Mean pairwise cosine of the (female - male) offset vectors.

    High -> the relation is a consistent translation, so additive `+/-` works.
    Low  -> the offsets are not parallel; `+/-` is a crude model of the relation.
    """
    offs = females - males
    n = len(offs)
    return float(np.mean([cos(offs[i], offs[j]) for i in range(n) for j in range(i + 1, n)]))


def loo_accuracy(males: np.ndarray, females: np.ndarray, dims: int = 6):
    """Leave-one-out top-1 accuracy for three relation operators.

    For each held-out pair k, fit the operator on the other pairs, predict the
    female from the held-out male, and check whether the true female is the
    nearest of all females. Fitting happens in a PCA-reduced space so the
    rotation/linear map are not hopelessly over-parameterized for so few pairs.
    (Illustrative: the tiny sample and the slight PCA leakage make these noisy.)
    """
    n = len(males)
    allX = np.vstack([males, females])
    mu = allX.mean(0)
    _, _, Vt = svd(allX - mu, full_matrices=False)
    basis = Vt[:dims]
    M = (males - mu) @ basis.T
    F = (females - mu) @ basis.T

    def is_top1(pred: np.ndarray, true_idx: int) -> bool:
        order = sorted(range(n), key=lambda j: cos(pred, F[j]), reverse=True)
        return order[0] == true_idx

    add = rot = lin = 0
    for k in range(n):
        tr = [i for i in range(n) if i != k]
        Mt, Ft = M[tr], F[tr]
        add += is_top1(M[k] + np.mean(Ft - Mt, axis=0), k)        # additive offset
        U, _, Vt2 = svd(Mt.T @ Ft)                                # orthogonal rotation
        rot += is_top1(M[k] @ (U @ Vt2), k)                       #   (Procrustes)
        W, _, _, _ = lstsq(Mt, Ft, rcond=None)                    # learned linear map
        lin += is_top1(M[k] @ W, k)
    return add, rot, lin


def analogy(embed_of: dict, a: str, b: str, c: str, candidates) -> list:
    """Rank candidates for the analogy b - a + c (e.g., king - man + woman)."""
    target = embed_of[b] - embed_of[a] + embed_of[c]
    return sorted(candidates, key=lambda w: cos(target, embed_of[w]), reverse=True)


def static_word2vec_control() -> None:
    """Positive control: classic analogies on GloVe (static word vectors)."""
    try:
        import gensim.downloader as api
    except ImportError:
        print("  [skipped] gensim not installed — run `pip install gensim` for the control.")
        return

    print("  loading GloVe (glove-wiki-gigaword-100; ~128 MB on first run, then cached)...")
    kv = api.load("glove-wiki-gigaword-100")

    # Standard analogy evaluation: gensim excludes the input words automatically.
    print("  analogies (inputs excluded — the standard, honest evaluation):")
    for a, b, c, expected in [("man", "king", "woman", "queen"),
                              ("france", "paris", "japan", "tokyo"),
                              ("walking", "walked", "swimming", "swam")]:
        word, score = kv.most_similar(positive=[b, c], negative=[a], topn=1)[0]
        mark = "OK" if word == expected else "  "
        print(f"    [{mark}] {b} - {a} + {c}  ->  {word} ({score:.3f})   [expected {expected}]")

    # The caveat (Drozd 2016; Nissim 2020): drop the exclusion and the nearest
    # vector is usually an input word itself, not "queen".
    target = kv["king"] - kv["man"] + kv["woman"]
    nearest = [w for w, _ in kv.similar_by_vector(target, topn=4)]
    print(f"  WITHOUT excluding inputs, nearest to king - man + woman = {nearest}")
    print("    -> an input word ('king') usually still ranks first; the clean result")
    print("       depends on the exclusion rule.")

    # Offset parallelism on the same gender pairs, to compare against GenAI Studio.
    have = [(m, f) for m, f in PAIRS if m in kv and f in kv]
    males = np.array([kv[m] for m, _ in have])
    females = np.array([kv[f] for _, f in have])
    print(f"  gender-offset parallelism: {offset_parallelism(males, females):+.3f}"
          f"  (static vectors -> strongly parallel, so `+/-` works)")


def contextual_genai_studio() -> None:
    """The same arithmetic on contextual embeddings from GenAI Studio."""
    if not os.environ.get("GENAI_STUDIO_API_KEY"):
        print("  [skipped] GENAI_STUDIO_API_KEY not set — see 01_first_chat.py.")
        return

    ai = GenAIStudio()
    words = sorted({w for pair in PAIRS for w in pair})
    n = len(PAIRS)

    for model in COMPARE_MODELS:
        try:
            ai.select_model(model)
            vecs = dict(zip(words, [np.array(v, float) for v in ai.embed(words)]))
        except Exception as exc:  # model may not expose embeddings
            print(f"\n  [{model}] skipped ({str(exc)[:60]})")
            continue

        males = np.array([vecs[m] for m, _ in PAIRS])
        females = np.array([vecs[f] for _, f in PAIRS])
        add, rot, lin = loo_accuracy(males, females)

        print(f"\n  [{model}]")
        if model == COMPARE_MODELS[0]:
            ranked = analogy(vecs, "man", "king", "woman",
                             [w for w in words if w not in {"man", "king", "woman"}])
            print(f"    king - man + woman  ->  {ranked[0]}  (next: {', '.join(ranked[1:4])})")
        print(f"    gender-offset parallelism: {offset_parallelism(males, females):+.3f}"
              f"  (contextual -> only weakly parallel)")
        print(f"    LOO top-1 (chance {1 / n:.0%}):  "
              f"additive={add}/{n}  rotation={rot}/{n}  linear-map={lin}/{n}")


def main() -> None:
    print("=" * 72)
    print("1) STATIC word vectors (Word2Vec/GloVe) — the classic positive control")
    print("=" * 72)
    static_word2vec_control()

    print("\n" + "=" * 72)
    print("2) CONTEXTUAL embeddings (GenAI Studio) — the same arithmetic, noisier")
    print("=" * 72)
    contextual_genai_studio()

    print("\n" + "=" * 72)
    print("Takeaway")
    print("=" * 72)
    print("Relationships ARE encoded geometrically, but `+/-` assumes a clean parallel")
    print("translation. Static word vectors satisfy that (offsets parallel, analogy")
    print("works); contextual embeddings only approximate it (offsets weakly parallel).")
    print("A rotation can match the additive offset, but learned maps overfit with few")
    print("pairs, and which operator wins varies by model (compare the two above) — there")
    print("is no single 'right' operator. See §6.2 and §6.9 Further Reading (superposition).")


if __name__ == "__main__":
    main()
