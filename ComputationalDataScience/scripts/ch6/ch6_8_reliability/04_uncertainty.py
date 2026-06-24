"""
STAT 418 · Chapter 6.8 — Reliability and Evaluation
Script 04: Uncertainty from self-consistency, with a bootstrap CI on agreement.

Running a prompt several times and measuring agreement gives a free, interpretable
uncertainty estimate: confident inputs converge, ambiguous ones spread out. We then
put a Chapter-4 bootstrap confidence interval on the agreement rate itself, so the
uncertainty estimate carries its own uncertainty.

Calls are spaced by RATE_LIMIT_DELAY to stay under the ~20 requests/minute limit.

Prereqs / run: see ../ch6_1_foundations/01_first_chat.py. (numpy.)
"""
from __future__ import annotations

import os
import sys
import time
from collections import Counter

import numpy as np

from genai_studio import GenAIStudio

CHAT_MODEL = "gemma3:12b"
RATE_LIMIT_DELAY = 3.5  # seconds between calls -> ~17 req/min, under the ~20/min limit

CLASSIFY_PROMPT = ("Classify this text as positive, negative, or neutral.\n"
                   "Respond with ONLY one word.\n\nText: {text}\nLabel:")


def classify(ai: GenAIStudio, text: str) -> str:
    try:
        return ai.chat(CLASSIFY_PROMPT.format(text=text)).strip().lower()
    except Exception:
        return "unparseable"


def uncertainty_via_consistency(ai: GenAIStudio, text: str, n_runs: int = 7) -> dict:
    """The agreement rate across n_runs is a natural, interpretable confidence."""
    answers = []
    for i in range(n_runs):
        if i:
            time.sleep(RATE_LIMIT_DELAY)
        answers.append(classify(ai, text))
    majority, count = Counter(answers).most_common(1)[0]
    return {"answer": majority, "confidence": count / n_runs,
            "uncertainty": 1 - count / n_runs, "distribution": dict(Counter(answers))}


def bootstrap_ci_on_agreement(answers, n_bootstrap=1000, alpha=0.05, seed=42):
    """Bootstrap CI on the majority-vote agreement rate (Chapter 4)."""
    rng = np.random.default_rng(seed)
    n = len(answers)
    majority = Counter(answers).most_common(1)[0][0]
    is_majority = np.array([1 if a == majority else 0 for a in answers])
    boot = [is_majority[rng.choice(n, size=n, replace=True)].mean() for _ in range(n_bootstrap)]
    return (float(np.percentile(boot, 100 * alpha / 2)),
            float(np.percentile(boot, 100 * (1 - alpha / 2))))


def make_client() -> GenAIStudio:
    if not os.environ.get("GENAI_STUDIO_API_KEY"):
        sys.exit("GENAI_STUDIO_API_KEY is not set - see ../ch6_1_foundations/01_first_chat.py.")
    ai = GenAIStudio()
    ai.select_model(CHAT_MODEL)
    return ai


def main() -> None:
    ai = make_client()

    print("Uncertainty via self-consistency (7 runs each):")
    for text in ["This is the best product I have ever purchased!",
                 "I love the quality but hate the price."]:
        r = uncertainty_via_consistency(ai, text, n_runs=7)
        print(f"  [{r['confidence']:>4.0%} conf] {r['answer']:>11} | {text[:46]}")
        print(f"    distribution: {r['distribution']}")

    # --- Bootstrap CI on a fixed set of runs (deterministic) ---
    answers = ["neutral", "negative", "neutral", "neutral", "negative",
               "neutral", "neutral", "negative", "negative", "neutral"]
    lo, hi = bootstrap_ci_on_agreement(answers)
    rate = Counter(answers).most_common(1)[0][1] / len(answers)
    print("\nBootstrap CI on agreement (10 fixed runs):")
    print(f"  agreement {rate:.0%}, 95% CI [{lo:.0%}, {hi:.0%}] (wide - only 10 runs)")


if __name__ == "__main__":
    main()
