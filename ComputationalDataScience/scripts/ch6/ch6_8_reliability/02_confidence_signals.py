"""
STAT 418 · Chapter 6.8 — Reliability and Evaluation
Script 02: Confidence signals — you cannot read a probability off an LLM.

A generative LLM answering free-form questions gives us no probability of being correct: the
token probabilities are not exposed, and a self-reported "confidence" is generated text, not a
measured probability. So we do NOT calibrate an LLM's raw answers. What we *can* build is a
behavior-based uncertainty heuristic — the self-consistency agreement rate — and then VALIDATE
it, because it measures stability, not correctness: a model can agree with itself on a wrong
answer (it is confidently wrong).

This script shows the agreement rate as a triage signal and checks it against ground truth.

Prereqs / run: see ../ch6_1_foundations/01_first_chat.py. (numpy.)
"""
from __future__ import annotations

import os
import sys
from collections import Counter

import numpy as np

from genai_studio import GenAIStudio
from genai_studio.agents import RateLimiter

CHAT_MODEL = "gemma3:12b"

# GenAI Studio caps at ~20 requests/min and SILENTLY drops bursts (no 429s);
# pace every gateway call through the SDK's RateLimiter.
RPM = 20
limiter = RateLimiter(RPM)

QA_PROMPT = ("Answer in as few words as possible. Give only the answer.\n\n"
             "Question: {question}\nAnswer:")


def agreement_confidence(ai: GenAIStudio, question: str, n_runs: int = 10):
    """Self-consistency: run n times, return (majority answer, agreement rate).

    The agreement rate is grounded in observed behavior — but it reports how STABLE
    the answer is, not whether it is RIGHT. (Needs temperature > 0 to vary; GenAI
    Studio's default sampling supplies it.)
    """
    answers = []
    for _ in range(n_runs):
        limiter.acquire()
        try:
            answers.append(ai.chat(QA_PROMPT.format(question=question)).strip().lower())
        except Exception:
            continue  # dropped/rate-limited run — skip it
    if not answers:
        return "", 0.0
    majority, k = Counter(answers).most_common(1)[0]
    return majority, k / len(answers)


def make_client() -> GenAIStudio:
    if not os.environ.get("GENAI_STUDIO_API_KEY"):
        sys.exit("GENAI_STUDIO_API_KEY is not set - see ../ch6_1_foundations/01_first_chat.py.")
    ai = GenAIStudio()
    ai.select_model(CHAT_MODEL)
    return ai


def main() -> None:
    ai = make_client()

    questions = [
        ("What is the capital of France?", "paris"),
        ("Who wrote Hamlet?", "shakespeare"),
        ("In what year did the Berlin Wall fall?", "1989"),
        ("What is the 12th prime number?", "37"),
    ]

    print(f"Self-consistency agreement ({CHAT_MODEL}) — stability, not correctness:")
    agree, correct = [], []
    for q, truth in questions:
        majority, conf = agreement_confidence(ai, q)
        hit = truth in majority
        agree.append(conf)
        correct.append(int(hit))
        print(f"  [{conf:>4.0%} agree] [{'OK' if hit else '  '}] {q} -> {majority[:20]}")

    agree, correct = np.array(agree), np.array(correct)
    high = agree >= 0.8
    if high.any():
        cw = int((high & (correct == 0)).sum())
        print(f"\n  high-agreement items (>=0.8): {high.sum()}, of which confidently WRONG: {cw}")
    print("  High agreement means the answer is STABLE, not that it is CORRECT.")
    print("  Use the agreement rate to triage (route low-agreement to a human) — never as a")
    print("  probability of being right, and always validate it against ground truth.")


if __name__ == "__main__":
    main()
