"""
STAT 418 · Chapter 6.4 — LLM-Assisted Data Annotation
Script 02: Common annotation tasks — fine-grained sentiment, NER, topic labels.

Once you can write a labeling prompt, the same pattern covers many tasks: a 1-5
sentiment rating, named-entity recognition with typed JSON output, and
single-label topic classification of headlines.

Prereqs / run: see ../ch6_1_foundations/01_first_chat.py for setup.
"""
from __future__ import annotations

import json
import os
import sys

from genai_studio import GenAIStudio

CHAT_MODEL = "gemma3:12b"

FINE_SENTIMENT_PROMPT = """Rate the sentiment of this review on a 1-5 scale:
1 = Very negative, 2 = Negative, 3 = Neutral, 4 = Positive, 5 = Very positive

Respond with ONLY the number (1, 2, 3, 4, or 5).

Review: {text}
Rating:"""

NER_PROMPT = """Extract all named entities from the following text.

Return a JSON list where each item has:
- "entity": the text of the entity
- "type": one of "PERSON", "ORGANIZATION", "LOCATION", "DATE"

Return ONLY the JSON list.

Text: {text}"""

TOPIC_PROMPT = """Classify this news headline into one of these categories:
sports, technology, politics, business, science, entertainment

Respond with ONLY the category name.

Headline: {text}
Category:"""


def annotate_ner(text: str, ai: GenAIStudio) -> list:
    """Extract typed named entities as a list of {entity, type} dicts."""
    try:
        response = ai.chat(NER_PROMPT.format(text=text)).strip()
    except Exception:
        return []
    if response.startswith("```"):                        # strip markdown fences
        response = response.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return []


def chat_label(prompt: str, ai: GenAIStudio) -> str:
    """Single-line label from a prompt; '?' on a dropped response."""
    try:
        return ai.chat(prompt).strip()
    except Exception:
        return "?"


def make_client() -> GenAIStudio:
    if not os.environ.get("GENAI_STUDIO_API_KEY"):
        sys.exit("GENAI_STUDIO_API_KEY is not set - see ../ch6_1_foundations/01_first_chat.py.")
    ai = GenAIStudio()
    ai.select_model(CHAT_MODEL)
    return ai


def main() -> None:
    ai = make_client()

    print(f"Fine-grained sentiment, 1-5 ({CHAT_MODEL}):")
    for text in ["Absolutely perfect, couldn't be happier!",
                 "It's fine, nothing special.",
                 "Completely useless, avoid at all costs."]:
        print(f"  [{chat_label(FINE_SENTIMENT_PROMPT.format(text=text), ai)}] {text}")

    print("\nNamed entity recognition:")
    entities = annotate_ner(
        "Dr. Smith from Purdue University visited Tokyo on March 15, 2024.", ai)
    for e in entities:
        print(f"  [{str(e.get('type', '?')):>12}] {e.get('entity', '?')}")

    print("\nTopic classification:")
    for headline in ["New AI chip doubles processing speed",
                     "Team wins championship in overtime",
                     "Senate passes infrastructure bill",
                     "Startup raises $50M in Series B",
                     "Researchers discover high-temperature superconductor"]:
        label = chat_label(TOPIC_PROMPT.format(text=headline), ai).lower()
        print(f"  [{label:>13}] {headline}")


if __name__ == "__main__":
    main()
