"""
STAT 418 · Chapter 6.4 — LLM-Assisted Data Annotation
Script 03: Quality control — Cohen's kappa, a bootstrap CI, and consensus voting.

LLM annotations are not ground truth; validate them with the Chapter 4 toolkit.
Cohen's kappa measures agreement corrected for chance (so a lazy "always
positive" annotator can't score well on an imbalanced set); the bootstrap
quantifies kappa's uncertainty; and consensus (majority vote over repeated runs)
improves reliability while flagging ambiguous items for human review.

The kappa + bootstrap demo uses a fixed simulated label set (deterministic, no
API). The consensus demo makes live calls and runs only if GENAI_STUDIO_API_KEY
is set.

Prereqs / run: see ../ch6_1_foundations/01_first_chat.py. (scikit-learn, numpy.)
"""
from __future__ import annotations

import os
import time
from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score

from genai_studio import GenAIStudio

RATE_LIMIT_DELAY = 3.5  # seconds between calls -> ~17 req/min, under the ~20/min limit

# Simulated LLM-vs-human labels for 50 reviews (book example): 42/50 agree.
HUMAN_LABELS = ["pos"] * 20 + ["neg"] * 15 + ["neu"] * 15
LLM_LABELS = (["pos"] * 18 + ["neg"] * 2 +
              ["neg"] * 12 + ["pos"] * 1 + ["neu"] * 2 +
              ["neu"] * 12 + ["pos"] * 1 + ["neg"] * 2)

CONSENSUS_PROMPT = ("Classify the sentiment as ONLY one word - positive, negative, "
                    "or neutral.\n\nReview: {text}\nLabel:")


def interpret_kappa(kappa: float) -> str:
    if kappa > 0.8:
        return "almost perfect agreement"
    if kappa > 0.6:
        return "substantial agreement"
    if kappa > 0.4:
        return "moderate agreement"
    return "fair or poor agreement"


def bootstrap_kappa(human_labels, llm_labels, n_bootstrap=1000, seed=42) -> dict:
    """Bootstrap confidence interval for Cohen's kappa (Chapter 4)."""
    rng = np.random.default_rng(seed)
    n = len(human_labels)
    kappas = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        h = [human_labels[i] for i in idx]
        l = [llm_labels[i] for i in idx]
        try:
            kappas.append(cohen_kappa_score(h, l))
        except ValueError:
            continue
    return {
        "mean_kappa": float(np.mean(kappas)),
        "ci_95": (float(np.percentile(kappas, 2.5)), float(np.percentile(kappas, 97.5))),
        "se": float(np.std(kappas)),
    }


def annotate_consensus(text, annotate_fn, ai, n_runs=5) -> dict:
    """Run annotation n_runs times; return the majority label + agreement fraction.

    Calls are spaced by RATE_LIMIT_DELAY to respect the ~20 requests/minute limit.
    """
    labels = []
    for r in range(n_runs):
        if r:
            time.sleep(RATE_LIMIT_DELAY)
        labels.append(annotate_fn(text, ai))
    label, count = Counter(labels).most_common(1)[0]
    return {"label": label, "agreement": count / n_runs, "all_labels": labels}


def main() -> None:
    # --- Cohen's kappa vs. raw accuracy (deterministic simulated labels) ---
    acc = accuracy_score(HUMAN_LABELS, LLM_LABELS)
    kappa = cohen_kappa_score(HUMAN_LABELS, LLM_LABELS)
    print(f"Accuracy:      {acc:.3f}")
    print(f"Cohen's kappa: {kappa:.3f}  ({interpret_kappa(kappa)})")
    print("  (kappa < accuracy because it corrects for chance agreement)")

    # --- Bootstrap CI on kappa (Chapter 4) ---
    result = bootstrap_kappa(HUMAN_LABELS, LLM_LABELS)
    print(f"\nBootstrap kappa: {result['mean_kappa']:.3f}, "
          f"95% CI [{result['ci_95'][0]:.3f}, {result['ci_95'][1]:.3f}], "
          f"SE {result['se']:.3f}")

    # --- Consensus via repeated runs (needs API; outputs are stochastic) ---
    if not os.environ.get("GENAI_STUDIO_API_KEY"):
        print("\n[consensus demo skipped - set GENAI_STUDIO_API_KEY to run it]")
        return
    ai = GenAIStudio()
    ai.select_model("gemma3:12b")

    def annotate(text, ai):
        try:
            r = ai.chat(CONSENSUS_PROMPT.format(text=text)).strip().lower()
        except Exception:
            return "unparseable"
        return r if r in {"positive", "negative", "neutral"} else "unparseable"

    print("\nConsensus over 5 runs (low agreement flags ambiguous items):")
    for text in ["The product is decent but overpriced.",
                 "Absolutely fantastic, highly recommend!"]:
        c = annotate_consensus(text, annotate, ai, n_runs=5)
        print(f"  {c['label']:>11} (agreement {c['agreement']:.0%}): {text}")
        print(f"              runs: {c['all_labels']}")


if __name__ == "__main__":
    main()
