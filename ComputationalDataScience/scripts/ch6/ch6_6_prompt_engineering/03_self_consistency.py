"""
STAT 418 · Chapter 6.6 — Prompt Engineering for Data Science
Script 03: Self-consistency — majority vote over multiple reasoning paths.

Self-consistency (Wang et al., 2022) runs the same prompt several times and takes
the majority answer. It is the prompting analogue of the Chapter 4 bootstrap: just
as bootstrap resamples data to gauge a statistic's variability, self-consistency
resamples reasoning paths to gauge an answer's reliability. High agreement is like
a narrow CI (trustworthy); low agreement is like a wide one (treat with caution).

Calls are spaced by RATE_LIMIT_DELAY to stay under the ~20 requests/minute limit,
so a 7-run pass takes about 25 seconds.

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


def self_consistency(ai: GenAIStudio, prompt: str, n_runs: int = 7) -> dict | None:
    """Run prompt n_runs times, extract the text after 'ANSWER:', and majority-vote."""
    answers = []
    for i in range(n_runs):
        if i:
            time.sleep(RATE_LIMIT_DELAY)
        try:
            response = ai.chat(prompt)
        except Exception:
            continue  # dropped/rate-limited run - just skip it
        if "ANSWER:" in response:
            answers.append(response.split("ANSWER:")[-1].strip().splitlines()[0].strip())
    if not answers:
        return None
    label, count = Counter(answers).most_common(1)[0]
    return {"answer": label, "agreement": count / len(answers),
            "n_valid": len(answers), "all": answers}


def make_client() -> GenAIStudio:
    if not os.environ.get("GENAI_STUDIO_API_KEY"):
        sys.exit("GENAI_STUDIO_API_KEY is not set - see ../ch6_1_foundations/01_first_chat.py.")
    ai = GenAIStudio()
    ai.select_model(CHAT_MODEL)
    return ai


def main() -> None:
    ai = make_client()
    prompt = ("A bag contains 3 red balls and 5 blue balls. You draw 2 balls without "
              "replacement. What is the probability that both are red?\n\n"
              "Think step by step, then give your final answer as a fraction.\n"
              "End with: ANSWER: [fraction]")

    print("Running self-consistency (7 reasoning paths, ~3.5s apart)...")
    result = self_consistency(ai, prompt, n_runs=7)
    if result is None:
        print("  no parseable answers (backend may be rate-limited; rerun).")
        return

    for i, ans in enumerate(result["all"], 1):
        print(f"  run {i}: {ans}")
    print(f"\nMajority answer: {result['answer']} "
          f"({result['agreement']:.0%} agreement over {result['n_valid']} valid runs)")
    print("  High agreement ~ a narrow bootstrap CI: the answer is robust.")
    print("  (correct: C(3,2)/C(8,2) = 3/28)")


if __name__ == "__main__":
    main()
