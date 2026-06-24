"""
STAT 418 · Chapter 6.8 — Reliability and Evaluation
Script 02: Calibration — does stated confidence match actual accuracy?

A model is well-calibrated if, across the predictions where it claims "80% sure,"
it is right about 80% of the time. We elicit a confidence with each answer, then
build a reliability diagram and compute the Expected Calibration Error (ECE). The
diagram runs on a fixed simulated set (deterministic, so the ECE is reproducible);
the confidence-elicitation demo uses the live model.

Prereqs / run: see ../ch6_1_foundations/01_first_chat.py. (numpy.)
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from genai_studio import GenAIStudio

CHAT_MODEL = "gemma3:12b"
RATE_LIMIT_DELAY = 3.5  # seconds between calls -> ~17 req/min, under the ~20/min limit

CALIBRATED_PROMPT = ("Answer the following question and rate your confidence.\n\n"
                     'Return a JSON object: {{"answer": "<your answer>", '
                     '"confidence": <0.0-1.0>}}\n'
                     "Confidence should reflect how likely your answer is correct.\n\n"
                     "Question: {question}")


def get_calibrated_answer(ai: GenAIStudio, question: str):
    """Elicit an answer plus a self-reported confidence (parsed from JSON)."""
    try:
        response = ai.chat(CALIBRATED_PROMPT.format(question=question)).strip()
    except Exception:
        return "", 0.5
    if response.startswith("```"):
        response = response.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        parsed = json.loads(response)
        return parsed.get("answer", ""), float(parsed.get("confidence", 0.5))
    except (json.JSONDecodeError, ValueError, TypeError):
        return response, 0.5


def reliability_diagram(confidences, correct, n_bins=10) -> dict:
    """Reliability-diagram bins + Expected Calibration Error (ECE).

    ECE is the count-weighted average gap between each bin's mean predicted
    confidence and its empirical accuracy. Lower is better; > 0.1 is poor.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_conf, bin_acc, bin_count = [], [], []
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if np.sum(mask) > 0:
            bin_conf.append(np.mean(confidences[mask]))
            bin_acc.append(np.mean(correct[mask]))
            bin_count.append(int(np.sum(mask)))
    ece = sum(c * abs(a - m) for c, a, m in zip(bin_count, bin_acc, bin_conf)) / sum(bin_count)
    return {"bin_conf": bin_conf, "bin_acc": bin_acc, "bin_count": bin_count, "ece": ece}


def make_client() -> GenAIStudio:
    if not os.environ.get("GENAI_STUDIO_API_KEY"):
        sys.exit("GENAI_STUDIO_API_KEY is not set - see ../ch6_1_foundations/01_first_chat.py.")
    ai = GenAIStudio()
    ai.select_model(CHAT_MODEL)
    return ai


def main() -> None:
    ai = make_client()

    # --- Confidence elicitation on factual questions (live) ---
    questions = [
        ("What is the capital of France?", "Paris"),
        ("What is 2 + 2?", "4"),
        ("Who wrote Hamlet?", "Shakespeare"),
        ("What year did World War I start?", "1914"),
    ]
    print(f"Confidence elicitation ({CHAT_MODEL}):")
    for i, (q, truth) in enumerate(questions):
        if i:
            time.sleep(RATE_LIMIT_DELAY)
        answer, conf = get_calibrated_answer(ai, q)
        mark = "OK" if truth.lower() in answer.lower() else "  "
        print(f"  [{conf:>4.0%} conf] [{mark}] {q} -> {answer[:30]}")

    # --- Reliability diagram + ECE on a fixed simulated set (deterministic) ---
    np.random.seed(42)
    confidences = np.random.beta(5, 2, 50)
    correct = np.random.binomial(1, confidences * 0.85 + 0.05)
    result = reliability_diagram(confidences, correct)
    print("\nReliability diagram on 50 simulated predictions:")
    print(f"  Expected Calibration Error (ECE): {result['ece']:.3f}")
    for c, a, n in zip(result["bin_conf"], result["bin_acc"], result["bin_count"]):
        tag = "overconfident" if c > a else "underconfident"
        print(f"    conf {c:.2f}: accuracy {a:.2f} (n={n}) [{tag}]")
    print("  ECE > 0.1 => significant miscalibration (this sim is deliberately so).")


if __name__ == "__main__":
    main()
