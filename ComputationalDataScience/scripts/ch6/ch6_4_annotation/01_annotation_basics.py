"""
STAT 418 · Chapter 6.4 — LLM-Assisted Data Annotation
Script 01: Annotation basics — turning a prompt into a labeling guideline.

The prompt IS the annotation guideline: clear categories, an output-format
constraint, and edge-case handling determine label quality. Three building
blocks: a single-label classifier (sentiment), structured JSON output
(sentiment + confidence + topics), and a batch pipeline with error tracking.

Annotation calls are wrapped so a malformed or dropped response becomes the
"unparseable" label rather than crashing a long batch.

Prereqs / run: see ../ch6_1_foundations/01_first_chat.py for setup.
"""
from __future__ import annotations

import json
import os
import sys
import time

from genai_studio import GenAIStudio

CHAT_MODEL = "gemma3:12b"
RATE_LIMIT_DELAY = 3.5  # seconds between calls -> ~17 req/min, under the ~20/min limit

SENTIMENT_PROMPT = """Classify the sentiment of the following customer review.

Categories:
- positive: The reviewer expresses satisfaction, approval, or recommendation
- negative: The reviewer expresses dissatisfaction, complaint, or warning
- neutral: The reviewer is factual or balanced without strong sentiment

Respond with ONLY the category label (positive, negative, or neutral).
Do not include any explanation.

Review: {text}
Label:"""

STRUCTURED_PROMPT = """Analyze the following customer review.

Return your response as a JSON object with these fields:
- "sentiment": one of "positive", "negative", "neutral"
- "confidence": a number from 0 to 1 indicating your confidence
- "topics": a list of topics mentioned (e.g., "quality", "shipping", "price")

Return ONLY the JSON object, no other text.

Review: {text}"""


def annotate_sentiment(text: str, ai: GenAIStudio) -> str:
    """Single-label sentiment; returns 'unparseable' on a bad or dropped response."""
    try:
        response = ai.chat(SENTIMENT_PROMPT.format(text=text)).strip().lower()
    except Exception:
        return "unparseable"
    return response if response in {"positive", "negative", "neutral"} else "unparseable"


def annotate_structured(text: str, ai: GenAIStudio) -> dict:
    """Structured (JSON) annotation: sentiment + confidence + topics."""
    try:
        response = ai.chat(STRUCTURED_PROMPT.format(text=text)).strip()
    except Exception:
        return {"sentiment": "unparseable", "confidence": 0, "topics": []}
    if response.startswith("```"):                        # strip markdown fences
        response = response.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {"sentiment": "unparseable", "confidence": 0, "topics": []}


def annotate_batch(texts, annotate_fn, ai, batch_name="batch"):
    """Annotate a batch, tracking progress and unparseable/dropped responses.

    Calls are spaced by RATE_LIMIT_DELAY to stay under GenAI Studio's ~20
    requests/minute limit; bursting past it gets requests dropped.
    """
    results, errors = [], 0
    for i, text in enumerate(texts):
        if i:
            time.sleep(RATE_LIMIT_DELAY)
        label = annotate_fn(text, ai)
        if label == "unparseable":
            errors += 1
        results.append({"text": text, "label": label, "index": i})
        if (i + 1) % 10 == 0:
            print(f"  {batch_name}: {i + 1}/{len(texts)} ({errors} errors)")
    print(f"  {batch_name}: complete. {len(texts)} items, "
          f"{errors} unparseable ({errors / len(texts):.1%})")
    return results


def make_client() -> GenAIStudio:
    if not os.environ.get("GENAI_STUDIO_API_KEY"):
        sys.exit("GENAI_STUDIO_API_KEY is not set - see ../ch6_1_foundations/01_first_chat.py.")
    ai = GenAIStudio()
    ai.select_model(CHAT_MODEL)
    return ai


def main() -> None:
    ai = make_client()

    print(f"Single-label sentiment ({CHAT_MODEL}):")
    for text in ["Absolutely love this product, best purchase ever!",
                 "Broke after two days, complete waste of money.",
                 "The package arrived on Tuesday as expected."]:
        print(f"  [{annotate_sentiment(text, ai):>11}] {text}")

    print("\nStructured JSON annotation:")
    result = annotate_structured(
        "Great quality but shipping took forever. Price was reasonable.", ai)
    print(json.dumps(result, indent=2))

    print("\nBatch annotation:")
    reviews = [
        "Excellent build quality!", "Terrible customer service.",
        "Works as described.", "Not worth the price.",
        "My kids love it!", "Arrived damaged.",
        "Perfect fit for my needs.", "Cheaply made.",
        "Fast shipping, great seller.", "Returning this immediately.",
        "Solid product overall.", "Disappointing quality.",
        "Highly recommend!", "Don't waste your money.",
        "Good value.", "Falls apart easily.",
        "Beautiful design.", "Nothing special.",
        "Exceeded expectations.", "Regret buying this.",
    ]
    annotate_batch(reviews, annotate_sentiment, ai, "Sentiment")


if __name__ == "__main__":
    main()
