"""
STAT 418 · Chapter 6.6 — Prompt Engineering for Data Science
Script 04: Iterating on prompts, and A/B testing them with a permutation test.

Two disciplines. First, systematic iteration: four versions of a sentiment prompt,
each fixing a specific failure mode (vague -> format constraint -> add neutral class
-> few-shot). Second, A/B testing with statistical rigor: run two prompts on a
labeled test set and use a *paired* permutation test (Chapter 4) to ask whether the
accuracy difference is real or noise. The test is paired because both prompts are
scored on the same texts (the book's Exercise 6.6.5 notes this is the correct choice).

Calls are spaced by RATE_LIMIT_DELAY to stay under the ~20 requests/minute limit.

Prereqs / run: see ../ch6_1_foundations/01_first_chat.py. (numpy.)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

from genai_studio import GenAIStudio

CHAT_MODEL = "gemma3:12b"
RATE_LIMIT_DELAY = 3.5  # seconds between calls -> ~17 req/min, under the ~20/min limit


def chat(ai: GenAIStudio, prompt: str) -> str:
    try:
        return ai.chat(prompt).strip()
    except Exception:
        return ""  # dropped/rate-limited - counts as an incorrect classification


def paired_permutation_p(a, b, n_perm=5000, seed=42) -> float:
    """Paired (sign-flip) permutation test on the per-item accuracy difference."""
    rng = np.random.default_rng(seed)
    diffs = np.asarray(b, float) - np.asarray(a, float)
    observed = diffs.mean()
    null = np.empty(n_perm)
    for i in range(n_perm):
        signs = rng.choice([1.0, -1.0], size=len(diffs))
        null[i] = (signs * diffs).mean()
    return float(np.mean(np.abs(null) >= abs(observed)))


def make_client() -> GenAIStudio:
    if not os.environ.get("GENAI_STUDIO_API_KEY"):
        sys.exit("GENAI_STUDIO_API_KEY is not set - see ../ch6_1_foundations/01_first_chat.py.")
    ai = GenAIStudio()
    ai.select_model(CHAT_MODEL)
    return ai


def main() -> None:
    ai = make_client()

    # --- Systematic iteration: each version targets a specific failure mode ---
    v1 = "Is this review positive or negative? Review: {text}"
    v2 = ("Classify the sentiment. Respond with exactly one word: positive or "
          "negative.\n\nReview: {text}\nSentiment:")
    v3 = ("Classify the sentiment. Respond with exactly one word: positive, "
          "negative, or neutral.\n\nReview: {text}\nSentiment:")
    v4 = """Classify the sentiment of each review.
Respond with exactly one word: positive, negative, or neutral.

Review: "Loved it, will buy again!"
Sentiment: positive

Review: "Awful quality, broke after one day."
Sentiment: negative

Review: "It works fine, nothing special."
Sentiment: neutral

Review: "{text}"
Sentiment:"""

    test_text = "The product is okay but shipping took forever"
    print("Prompt iteration (same input, four versions):")
    for i, (label, template) in enumerate([("v1", v1), ("v2", v2), ("v3", v3), ("v4", v4)]):
        if i:
            time.sleep(RATE_LIMIT_DELAY)
        print(f"  {label}: {chat(ai, template.format(text=test_text))[:55]}")

    # --- A/B test two topic-classification prompts ---
    prompt_a = ("Classify the topic. Choose from: technology, politics, sports, "
                "science, entertainment. One word only.\n\nText: {text}\nTopic:")
    prompt_b = """You are a content categorizer. Assign the text to one category.
Categories: technology, politics, sports, science, entertainment.

Text: {text}
Category (one word):"""
    test_cases = [
        ("NASA's Perseverance rover collected its first Mars rock sample", "science"),
        ("Apple unveiled the new iPhone at its annual keynote", "technology"),
        ("The Lakers defeated the Celtics in overtime", "sports"),
        ("Congress passed a bipartisan infrastructure bill", "politics"),
        ("The new Marvel movie broke box office records", "entertainment"),
    ]
    print("\nA/B test (topic classification):")
    a_scores, b_scores = [], []
    for i, (text, truth) in enumerate(test_cases):
        if i:
            time.sleep(RATE_LIMIT_DELAY)
        a_scores.append(int(truth in chat(ai, prompt_a.format(text=text)).lower()))
        time.sleep(RATE_LIMIT_DELAY)
        b_scores.append(int(truth in chat(ai, prompt_b.format(text=text)).lower()))

    acc_a, acc_b = np.mean(a_scores), np.mean(b_scores)
    p = paired_permutation_p(a_scores, b_scores)
    print(f"  Prompt A accuracy: {acc_a:.0%}")
    print(f"  Prompt B accuracy: {acc_b:.0%}")
    print(f"  paired permutation p-value: {p:.3f} "
          f"-> {'significant' if p < 0.05 else 'no significant difference'} at alpha=0.05")
    print("  (With only 5 cases there is little power; 20+ are needed to detect")
    print("   modest differences - the point of the exercise.)")


if __name__ == "__main__":
    main()
