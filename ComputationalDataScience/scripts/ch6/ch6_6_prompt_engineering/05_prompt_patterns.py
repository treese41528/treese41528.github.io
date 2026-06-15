"""
STAT 418 · Chapter 6.6 — Prompt Engineering for Data Science
Script 05: Prompt chaining, multi-turn conversations, and data-science patterns.

Complex tasks are easier as a sequence of simple prompts: here we chain extraction
-> analysis. The Conversation class keeps context across turns. And two recurring
data-science patterns: a structured dataset description and code generation (always
with the reminder to verify any generated code before trusting it).

Prereqs / run: see ../ch6_1_foundations/01_first_chat.py for setup.
"""
from __future__ import annotations

import os
import sys

from genai_studio import Conversation, GenAIStudio

CHAT_MODEL = "gemma3:12b"


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

    # --- Prompt chaining: the output of step 1 becomes the input to step 2 ---
    text = ("The study enrolled 150 patients with Type 2 diabetes across 3 sites. "
            "After 12 weeks, HbA1c decreased from 8.5% to 7.2% in the treatment group "
            "and from 8.4% to 8.1% in the control group.")
    extracted = chat(ai, f"Extract sample size, duration, treatment result, and "
                     f"control result as a numbered list.\n\n<text>\n{text}\n</text>")
    print(f"Step 1 - extraction:\n{extracted[:280]}")
    analysis = chat(ai, f"Given these study results:\n{extracted}\n\nCompute the "
                    f"difference-in-differences and say whether it is a clinically "
                    f"meaningful HbA1c reduction. Under 80 words.")
    print(f"\nStep 2 - analysis:\n{analysis[:280]}")

    # --- Multi-turn conversation (chat_conversation auto-tracks the reply) ---
    print("\nMulti-turn conversation:")
    conv = Conversation(system="You are a concise data science consultant.")
    conv.add_user("I have 50,000 customer reviews to categorize by topic. Approach?")
    try:
        r1 = ai.chat_conversation(conv)
        print(f"  Turn 1: {r1.content[:160]}...")
        conv.add_user("How would I evaluate the categorization quality?")
        r2 = ai.chat_conversation(conv)
        print(f"  Turn 2: {r2.content[:160]}...")
    except Exception as exc:
        print(f"  [conversation dropped: {exc}]")

    # --- DS pattern: structured dataset description ---
    desc = ("DataFrame with 5000 rows: customer_id (int), purchase_date (datetime), "
            "amount (float, 0.99-4999.99), category (str, 12 values), "
            "return_flag (bool, 8% True)")
    summary = chat(ai, f"Describe this dataset: {desc}\n\nGive the unit of "
                   f"observation and three analysis questions it could answer.")
    print(f"\nDataset description:\n{summary[:280]}")

    # --- DS pattern: code generation (review before trusting!) ---
    code = chat(ai, "Write a Python function percentile_ci(data, stat=np.mean, "
                "n_boot=10000, conf=0.95, seed=42) returning a bootstrap percentile "
                "CI as (point, lo, hi). Type hints + one-line docstring. Code only.")
    print(f"\nGenerated code (VERIFY before using):\n{code[:280]}")


if __name__ == "__main__":
    main()
