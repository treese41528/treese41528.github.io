"""
STAT 418 · Chapter 6.6 — Prompt Engineering for Data Science
Script 01: Prompts as code — parameterized templates, versioning, and structure.

A prompt used at scale IS your algorithm, so treat it like code: parameterize it,
version it, and constrain its output. We show a versioned template registry,
XML-style delimiters that separate instructions from data, and a strict JSON
output format with fence-tolerant parsing.

Prereqs / run: see ../ch6_1_foundations/01_first_chat.py for setup.
"""
from __future__ import annotations

import json
import os
import sys

from genai_studio import GenAIStudio

CHAT_MODEL = "gemma3:12b"

# A versioned template registry - prompts live in code, with version + notes.
PROMPTS = {
    "sentiment": {
        "version": "1.2",
        "template": ("Classify the sentiment of the following text.\n"
                     "Respond with exactly one word: positive, negative, or neutral.\n\n"
                     "Text: {text}\nSentiment:"),
        "model": "gemma3:12b",
        "notes": "v1.2: added 'exactly one word' constraint to reduce verbosity",
    },
}


def chat(ai: GenAIStudio, prompt: str) -> str:
    try:
        return ai.chat(prompt).strip()
    except Exception as exc:
        return f"[call dropped: {exc}]"


def parse_json(raw: str):
    """Parse a JSON response, tolerating ```json ... ``` markdown fences."""
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return {"error": "unparseable", "raw_preview": raw[:80]}


def make_client() -> GenAIStudio:
    if not os.environ.get("GENAI_STUDIO_API_KEY"):
        sys.exit("GENAI_STUDIO_API_KEY is not set - see ../ch6_1_foundations/01_first_chat.py.")
    ai = GenAIStudio()
    ai.select_model(CHAT_MODEL)
    return ai


def main() -> None:
    ai = make_client()

    # --- Parameterized, versioned template ---
    cfg = PROMPTS["sentiment"]
    out = chat(ai, cfg["template"].format(text="The service was terrible."))
    print(f"Versioned template [{cfg['version']}] -> {out}")

    # --- Delimiters separate instructions from data ---
    text = ("The company reported strong Q3 earnings, beating analyst expectations "
            "by 15%. However, forward guidance was cautious due to supply chain "
            "uncertainties.")
    delimited = (f"Analyze the following financial text for sentiment.\n\n"
                 f"<text>\n{text}\n</text>\n\n"
                 f"Format:\nOverall sentiment: [positive/negative/mixed]\n"
                 f"Key positive signals: [list]\nKey negative signals: [list]")
    print(f"\nDelimited prompt ->\n{chat(ai, delimited)[:300]}")

    # --- Strict JSON output format, parsed robustly ---
    abstract_prompt = """Extract structured information from this research abstract.

<abstract>
We ran a randomized controlled trial with 500 participants to evaluate a new
teaching method. The treatment group (n=250) showed a mean improvement of 12.3
points (SD=4.1) vs 8.7 (SD=3.8) in control. The difference was significant
(t=10.4, p<0.001, Cohen's d=0.93).
</abstract>

Return ONLY a JSON object with these fields:
- study_type: string
- sample_size: integer
- effect_size: number
- significant: boolean"""
    parsed = parse_json(chat(ai, abstract_prompt))
    print(f"\nStructured JSON extraction -> {parsed}")


if __name__ == "__main__":
    main()
