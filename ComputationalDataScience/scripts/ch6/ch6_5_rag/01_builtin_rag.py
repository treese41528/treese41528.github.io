"""
STAT 418 · Chapter 6.5 — Retrieval-Augmented Generation
Script 01: Built-in RAG via the GenAI Studio knowledge base API.

GenAI Studio handles chunking, embedding, storage, and retrieval for you: upload
a file, create a knowledge base, link them, then pass collections=[kb.id] to any
chat call. We compare a RAG-grounded answer against the same question asked
without RAG.

This script CREATES a knowledge base on your account (named below) and leaves it
in place for reuse (re-running reuses it rather than duplicating). It uploads the
bundled sample_syllabus.md. To remove it later: ai.delete_knowledge_base(kb.id).

Prereqs / run: see ../ch6_1_foundations/01_first_chat.py for setup.
"""
from __future__ import annotations

import os
import sys
import time

from genai_studio import GenAIStudio

CHAT_MODEL = "gemma3:12b"
KB_NAME = "STAT418-Ch6.5-Demo"
SAMPLE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_syllabus.md")


def get_or_create_kb(ai):
    """Reuse the demo KB if it already exists, else create it and link the file."""
    for kb in ai.list_knowledge_bases():
        if kb.name == KB_NAME:
            print(f"Reusing existing knowledge base: {kb.id}")
            return kb
    print(f"Uploading {os.path.basename(SAMPLE_FILE)} ...")
    file_info = ai.upload_file(SAMPLE_FILE)
    kb = ai.create_knowledge_base(KB_NAME, "STAT 418 demo syllabus for Chapter 6.5")
    ai.add_file_to_knowledge_base(kb.id, file_info.id)
    print(f"Created knowledge base {kb.id} and linked file {file_info.id}")
    time.sleep(3)  # give the backend a moment to index the file
    return kb


def ask(ai, question, **kwargs):
    try:
        return ai.chat(question, **kwargs)
    except Exception as exc:
        return f"[call dropped: {exc}]"


def make_client():
    if not os.environ.get("GENAI_STUDIO_API_KEY"):
        sys.exit("GENAI_STUDIO_API_KEY is not set - see ../ch6_1_foundations/01_first_chat.py.")
    ai = GenAIStudio()
    ai.select_model(CHAT_MODEL)
    return ai


def main():
    ai = make_client()
    kb = get_or_create_kb(ai)

    for question in ["What are the prerequisites for STAT 418?",
                     "What is the late homework policy?"]:
        print("\n" + "=" * 68)
        print(f"Q: {question}")
        print(f"\nWITHOUT RAG:\n  {ask(ai, question)[:240]}")
        print(f"\nWITH RAG (collections=[kb.id]):\n  {ask(ai, question, collections=[kb.id])[:240]}")

    print(f"\n(The '{KB_NAME}' knowledge base was left on your account; "
          f"remove it with ai.delete_knowledge_base('{kb.id}').)")


if __name__ == "__main__":
    main()
