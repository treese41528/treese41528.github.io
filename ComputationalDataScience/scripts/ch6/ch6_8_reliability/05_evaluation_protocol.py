"""
STAT 418 · Chapter 6.8 — Reliability and Evaluation
Script 05: A complete evaluation protocol — accuracy, F1, consistency, and a
deployment decision.

Ties the section together: build a labeled eval set, then measure accuracy and F1
(against ground truth) plus mean consistency (stability across repeated runs), and
check the numbers against explicit deployment thresholds. This is the discipline
that separates "the model seems fine" from "the model meets our bar."

Calls are spaced by RATE_LIMIT_DELAY and there are many of them
(items x (1 + n_consistency_runs)), so expect a couple of minutes.

Prereqs / run: see ../ch6_1_foundations/01_first_chat.py. (scikit-learn, numpy.)
"""
from __future__ import annotations

import os
import sys
import time
from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from genai_studio import GenAIStudio

CHAT_MODEL = "gemma3:12b"
RATE_LIMIT_DELAY = 3.5  # seconds between calls -> ~17 req/min, under the ~20/min limit

CLASSIFY_PROMPT = ("Classify this text as positive, negative, or neutral.\n"
                   "Respond with ONLY one word.\n\nText: {text}\nLabel:")

DEPLOYMENT_CRITERIA = {"min_accuracy": 0.80, "min_f1": 0.75, "min_consistency": 0.70}


def classify(ai: GenAIStudio, text: str) -> str:
    try:
        label = ai.chat(CLASSIFY_PROMPT.format(text=text)).strip().lower()
    except Exception:
        return "unparseable"
    return label if label in {"positive", "negative", "neutral"} else "unparseable"


def evaluate_model(ai: GenAIStudio, eval_data: list, n_consistency_runs: int = 2) -> dict:
    """Accuracy + weighted F1 (vs. ground truth) and mean consistency (stability)."""
    predictions, consistencies = [], []
    throttle = False
    for item in eval_data:
        if throttle:
            time.sleep(RATE_LIMIT_DELAY)
        throttle = True
        predictions.append(classify(ai, item["text"]))
        runs = []
        for _ in range(n_consistency_runs):
            time.sleep(RATE_LIMIT_DELAY)
            runs.append(classify(ai, item["text"]))
        consistencies.append(Counter(runs).most_common(1)[0][1] / n_consistency_runs)
    truth = [item["label"] for item in eval_data]
    return {"accuracy": accuracy_score(truth, predictions),
            "f1": f1_score(truth, predictions, average="weighted", zero_division=0),
            "consistency": float(np.mean(consistencies)),
            "predictions": predictions}


def make_client() -> GenAIStudio:
    if not os.environ.get("GENAI_STUDIO_API_KEY"):
        sys.exit("GENAI_STUDIO_API_KEY is not set - see ../ch6_1_foundations/01_first_chat.py.")
    ai = GenAIStudio()
    ai.select_model(CHAT_MODEL)
    return ai


def main() -> None:
    ai = make_client()
    texts = ["Great product!", "Terrible quality.", "It's fine.",
             "Love everything about it!", "Worst purchase ever.",
             "Standard quality.", "Highly recommend!", "Don't buy this.",
             "Nothing special.", "Perfect in every way."]
    labels = ["positive", "negative", "neutral", "positive", "negative",
              "neutral", "positive", "negative", "neutral", "positive"]
    eval_data = [{"text": t, "label": ll} for t, ll in zip(texts, labels)]

    print(f"Evaluating {len(eval_data)} items, 2 consistency runs each (~2 minutes)...")
    r = evaluate_model(ai, eval_data, n_consistency_runs=2)
    print(f"  accuracy:         {r['accuracy']:.3f}")
    print(f"  F1 (weighted):    {r['f1']:.3f}")
    print(f"  mean consistency: {r['consistency']:.3f}")

    print("\nDeployment readiness:")
    metrics = {"accuracy": r["accuracy"], "f1": r["f1"], "consistency": r["consistency"]}
    all_pass = True
    for crit, thresh in DEPLOYMENT_CRITERIA.items():
        name = crit.replace("min_", "")
        passed = metrics[name] >= thresh
        all_pass = all_pass and passed
        print(f"  {name:12s}: {metrics[name]:.3f} >= {thresh}  {'PASS' if passed else 'FAIL'}")
    print(f"\n  Decision: {'DEPLOY' if all_pass else 'DO NOT DEPLOY (review failures first)'}")


if __name__ == "__main__":
    main()
