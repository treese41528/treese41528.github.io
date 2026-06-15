"""
STAT 418 · Chapter 6.7 — Reliability and Evaluation
Script 01: Consistency — test-retest reliability and paraphrase invariance.

LLM generation is stochastic, so the same question can get different answers. Two
consistency checks: test-retest (ask the same thing N times, measure agreement)
and paraphrase invariance (reword the question, expect the same label). Low
agreement flags items for human review.

Calls are spaced by RATE_LIMIT_DELAY to stay under the ~20 requests/minute limit.

Prereqs / run: see ../ch6_1_foundations/01_first_chat.py for setup.
"""
from __future__ import annotations

import os
import sys
import time
from collections import Counter

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


def test_retest(ai: GenAIStudio, text: str, n_runs: int = 10) -> dict:
    """Ask the same question n_runs times and measure answer agreement."""
    answers = []
    for i in range(n_runs):
        if i:
            time.sleep(RATE_LIMIT_DELAY)
        answers.append(classify(ai, text))
    majority, count = Counter(answers).most_common(1)[0]
    return {"majority": majority, "agreement": count / n_runs,
            "unique": len(set(answers)), "answers": answers}


def paraphrase_invariance(ai: GenAIStudio, paraphrases: list) -> dict:
    """Check whether semantically equivalent inputs receive the same label."""
    labels = []
    for i, p in enumerate(paraphrases):
        if i:
            time.sleep(RATE_LIMIT_DELAY)
        labels.append(classify(ai, p))
    majority, count = Counter(labels).most_common(1)[0]
    return {"labels": labels, "majority": majority,
            "agreement": count / len(labels), "invariant": len(set(labels)) == 1}


def make_client() -> GenAIStudio:
    if not os.environ.get("GENAI_STUDIO_API_KEY"):
        sys.exit("GENAI_STUDIO_API_KEY is not set - see ../ch6_1_foundations/01_first_chat.py.")
    ai = GenAIStudio()
    ai.select_model(CHAT_MODEL)
    return ai


def main() -> None:
    ai = make_client()

    print("Test-retest reliability (same ambiguous text, 10 runs):")
    r = test_retest(ai, "The product is decent but a bit overpriced.", n_runs=10)
    print(f"  majority: {r['majority']} ({r['agreement']:.0%}), {r['unique']} unique answers")
    print(f"  answers: {r['answers']}")
    verdict = ("stable - trustworthy" if r["agreement"] >= 0.8
               else "low agreement - flag this item for human review (genuine uncertainty)")
    print(f"  -> {r['agreement']:.0%} agreement: {verdict}")

    print("\nParaphrase invariance (5 rewordings of the same sentiment):")
    paraphrases = [
        "The product quality is good but the price is too high.",
        "Good quality, though it's overpriced.",
        "Quality-wise it's fine, but I paid too much.",
        "Nice product, just wish it were cheaper.",
        "The quality is decent but the cost is steep.",
    ]
    pi = paraphrase_invariance(ai, paraphrases)
    print(f"  labels: {pi['labels']}")
    print(f"  agreement: {pi['agreement']:.0%}, invariant: {pi['invariant']}")
    verdict = ("invariant - robust to phrasing" if pi["invariant"]
               else "NOT invariant - rewording flips the label, a reliability problem")
    print(f"  -> {verdict}")


if __name__ == "__main__":
    main()
