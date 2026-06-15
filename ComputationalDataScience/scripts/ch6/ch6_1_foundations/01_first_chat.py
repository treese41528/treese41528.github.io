"""
STAT 418 · Chapter 6.1 — LLM Foundations
Script 01: Setting up the GenAI Studio client and your first chat.

Backs the §6.1 "Setup" and "First chat" slides. Demonstrates:
  - creating the client (reads GENAI_STUDIO_API_KEY from the environment),
  - listing and selecting models,
  - ai.chat()           -> the response as a plain string,
  - ai.chat_complete()  -> a rich object with token-count metadata.

Prerequisites
-------------
    pip install "git+https://github.com/treese41528/genai-studio-sdk.git@v1.2.0"
    export GENAI_STUDIO_API_KEY="sk-..."   # GenAI Studio -> Settings -> Account -> API Keys

Run
---
    python 01_first_chat.py

Note: LLM outputs are stochastic — your wording will differ from any example.
"""
from __future__ import annotations

import os
import sys

from genai_studio import GenAIStudio

MODEL = "gemma3:12b"


def setup_client() -> GenAIStudio:
    """Create the client and select a model."""
    if not os.environ.get("GENAI_STUDIO_API_KEY"):
        sys.exit(
            "GENAI_STUDIO_API_KEY is not set.\n"
            "Get a key at https://genai.rcac.purdue.edu (Settings -> Account -> API Keys), then:\n"
            '    export GENAI_STUDIO_API_KEY="sk-..."'
        )

    ai = GenAIStudio()

    print("Available models:")
    for model in ai.models:
        print(f"  - {model}")

    ai.select_model(MODEL)
    print(f"\nSelected model: {ai.model}")
    return ai


def first_chat(ai: GenAIStudio) -> None:
    """ai.chat(prompt) sends a prompt and returns the response as a string."""
    print("\n--- chat() ---")
    response = ai.chat("What is the Central Limit Theorem? Answer in two sentences.")
    print(response)


def chat_with_metadata(ai: GenAIStudio) -> None:
    """ai.chat_complete(prompt) returns a response object with token counts."""
    print("\n--- chat_complete() (with token metadata) ---")
    response = ai.chat_complete(
        "Explain the difference between correlation and causation in one paragraph."
    )
    print(response.content)
    print(f"\nModel: {response.model}")
    print(f"Finish reason: {response.finish_reason}")
    print(
        "Prompt / completion / total tokens: "
        f"{response.prompt_tokens} / {response.completion_tokens} / {response.total_tokens}"
    )


def main() -> None:
    ai = setup_client()
    first_chat(ai)
    chat_with_metadata(ai)


if __name__ == "__main__":
    main()
