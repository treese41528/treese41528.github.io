"""
STAT 418 · Chapter 6.1 — LLM Foundations
Script 02: Streaming, model comparison, system messages, multi-turn chat.

Backs the §6.1 "GenAI Studio essentials" slide. Demonstrates:
  - ai.chat_stream()         -> tokens as they are generated,
  - comparing models on one prompt with ai.chat_complete(),
  - ai.chat(..., system=...) -> a system message sets persona/behavior,
  - the Conversation class + ai.chat_conversation() for multi-turn dialogue.

Prerequisites: see 01_first_chat.py (install the pinned SDK, set GENAI_STUDIO_API_KEY).

Run
---
    python 02_streaming_system_multiturn.py
"""
from __future__ import annotations

import os
import sys

from genai_studio import GenAIStudio, Conversation

DEFAULT_MODEL = "gemma3:12b"


def make_client() -> GenAIStudio:
    if not os.environ.get("GENAI_STUDIO_API_KEY"):
        sys.exit("GENAI_STUDIO_API_KEY is not set — see 01_first_chat.py for setup.")
    ai = GenAIStudio()
    ai.select_model(DEFAULT_MODEL)
    return ai


def streaming(ai: GenAIStudio) -> None:
    """chat_stream() yields chunks as the model generates them."""
    print("--- streaming ---")
    for chunk in ai.chat_stream("List three assumptions of linear regression."):
        print(chunk, end="", flush=True)
    print()


def compare_models(ai: GenAIStudio) -> None:
    """Same prompt across models — pick the right tool for each task."""
    print("\n--- comparing models ---")
    prompt = "What is a p-value? Explain simply in one sentence."
    for model_name in ("mistral:latest", "gemma3:12b", "llama3.2:latest"):
        ai.select_model(model_name)
        resp = ai.chat_complete(prompt)
        print(f"[{model_name}] ({resp.total_tokens} tokens)")
        print(f"  {resp.content}\n")
    ai.select_model(DEFAULT_MODEL)


def system_message(ai: GenAIStudio) -> None:
    """A system message sets the model's persona/behavior for the turn."""
    print("--- system message ---")
    system_msg = (
        "You are a statistics professor at Purdue University. "
        "You explain concepts precisely, using mathematical notation where "
        "appropriate. Keep answers concise."
    )
    response = ai.chat(
        "What is the difference between a parameter and a statistic?",
        system=system_msg,
    )
    print(response)


def multi_turn(ai: GenAIStudio) -> None:
    """The Conversation class tracks history so follow-ups have context."""
    print("\n--- multi-turn conversation ---")
    conv = Conversation(system="You are a helpful data science tutor.")

    conv.add_user("What is overfitting?")
    response = ai.chat_conversation(conv)
    print(f"Assistant: {response.content}\n")

    conv.add_user("How does cross-validation help prevent it?")
    response = ai.chat_conversation(conv)
    print(f"Assistant: {response.content}")


def main() -> None:
    ai = make_client()
    streaming(ai)
    compare_models(ai)
    system_message(ai)
    multi_turn(ai)


if __name__ == "__main__":
    main()
