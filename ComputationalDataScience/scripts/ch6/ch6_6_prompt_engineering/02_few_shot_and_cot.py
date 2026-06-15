"""
STAT 418 · Chapter 6.6 — Prompt Engineering for Data Science
Script 02: Few-shot prompting and chain-of-thought reasoning.

Two high-leverage techniques. Few-shot prompting shows the model a few labeled
examples so it infers the pattern. Chain-of-thought (CoT) asks the model to reason
step by step before answering - which on multi-step problems can flip a wrong
direct answer into a right one and lets you verify each step.

Prereqs / run: see ../ch6_1_foundations/01_first_chat.py for setup.
"""
from __future__ import annotations

import os
import sys

from genai_studio import GenAIStudio

CHAT_MODEL = "gemma3:12b"

FEW_SHOT = """Classify each research method as quantitative, qualitative, or mixed methods.

Example 1:
Method: Survey with Likert-scale responses analyzed using factor analysis
Classification: quantitative

Example 2:
Method: Semi-structured interviews analyzed with thematic coding
Classification: qualitative

Example 3:
Method: Survey with open-ended responses, analyzed with both regression and content analysis
Classification: mixed methods

Now classify:
Method: Randomized controlled trial with 200 participants measuring blood pressure changes
Classification:"""

STATS_QUESTION = ("A researcher collects 50 samples from a population with unknown mean. "
                  "The sample mean is 23.4 and the sample standard deviation is 5.2. Is "
                  "there evidence that the population mean differs from 25 at the 5% "
                  "significance level?")


def chat(ai: GenAIStudio, prompt: str) -> str:
    try:
        return ai.chat(prompt).strip()
    except Exception as exc:
        return f"[call dropped: {exc}]"


def make_client() -> GenAIStudio:
    if not os.environ.get("GENAI_STUDIO_API_KEY"):
        sys.exit("GENAI_STUDIO_API_KEY is not set - see ../ch6_1_foundations/01_first_chat.py.")
    ai = GenAIStudio()
    ai.select_model(CHAT_MODEL)
    return ai


def main() -> None:
    ai = make_client()

    # --- Few-shot: infer the pattern from three labeled examples ---
    print(f"Few-shot classification -> {chat(ai, FEW_SHOT)}")

    # --- Chain-of-thought: a direct answer can be wrong where CoT is right ---
    direct = chat(ai, STATS_QUESTION + " Answer yes or no.")
    print(f"\nDirect answer -> {direct[:80]}")
    cot = chat(ai, STATS_QUESTION + "\n\nThink step by step: state the hypotheses, "
               "compute the test statistic, compare to the critical value, conclude. "
               "Show your work.")
    print(f"\nChain-of-thought ->\n{cot[:420]}")

    # --- Zero-shot CoT: just append the magic phrase ---
    zs = chat(ai, "A dataset has 1000 observations. You use 5-fold cross-validation "
              "with a model that takes 2 seconds to train. How long will the entire "
              "cross-validation take?\n\nLet's think step by step.")
    print(f"\nZero-shot CoT ->\n{zs[:300]}")


if __name__ == "__main__":
    main()
