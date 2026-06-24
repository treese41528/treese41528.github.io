"""
STAT 418 · Chapter 6.8 — Reliability and Evaluation
Script 03: LLM-as-judge — scoring outputs when there is no ground truth.

For open-ended tasks (summaries, explanations) there is often no gold label, so one
LLM can score another against a rubric. Useful, but judges have known biases - they
favor longer, same-style answers and tend to be generous - so treat the score as
one signal among several, not the final word.

Prereqs / run: see ../ch6_1_foundations/01_first_chat.py for setup.
"""
from __future__ import annotations

import os
import sys

from genai_studio import GenAIStudio

CHAT_MODEL = "gemma3:12b"

JUDGE_PROMPT = """You are an evaluation assistant. Score the following response
on a scale of 1-5 based on this rubric:

5: Completely accurate, well-organized, addresses all aspects
4: Mostly accurate with minor omissions
3: Partially accurate, missing important details
2: Contains significant errors or misunderstandings
1: Largely incorrect or irrelevant

### Question ###
{question}

### Reference Answer ###
{reference}

### Response to Evaluate ###
{response}

Score (1-5):"""


def judge_response(ai: GenAIStudio, question: str, reference: str, response: str) -> int:
    """Return an integer 1-5 score from the judge model (defaults to 3 on a parse fail)."""
    try:
        score_text = ai.chat(JUDGE_PROMPT.format(
            question=question, reference=reference, response=response)).strip()
    except Exception:
        return 3
    for ch in score_text:          # first digit 1-5 anywhere in the reply
        if ch in "12345":
            return int(ch)
    return 3


def make_client() -> GenAIStudio:
    if not os.environ.get("GENAI_STUDIO_API_KEY"):
        sys.exit("GENAI_STUDIO_API_KEY is not set - see ../ch6_1_foundations/01_first_chat.py.")
    ai = GenAIStudio()
    ai.select_model(CHAT_MODEL)
    return ai


def main() -> None:
    ai = make_client()
    question = "What is the bootstrap method?"
    reference = ("The bootstrap resamples observed data with replacement to estimate "
                 "the sampling distribution of a statistic.")
    candidates = {
        "strong": ("The bootstrap is a statistical technique that creates multiple "
                   "datasets by randomly sampling with replacement. It helps estimate "
                   "confidence intervals."),
        "weak": "The bootstrap is a way to start a computer when you turn it on.",
    }
    print(f"LLM-as-judge scoring ({CHAT_MODEL}):")
    for label, response in candidates.items():
        print(f"  {label:>6} response -> {judge_response(ai, question, reference, response)}/5")
    print("  (a good judge separates the two; remember judges skew long and generous.)")


if __name__ == "__main__":
    main()
