"""
STAT 418 · Chapter 6.3 — Text Preprocessing for LLM Pipelines
Script 01: Counting tokens — the unit everything in an LLM pipeline is measured in.

Context windows, API pricing, and embedding limits are all denominated in
*tokens*, not words. GenAI Studio doesn't expose its tokenizer, so we (a) estimate
tokens with simple heuristics and (b) measure actual usage from chat_complete()
metadata. We then compare the token/word ratio across content types — prose,
code, URLs, math, non-English — to see why one heuristic isn't enough (Ex. 6.3.1).

Prereqs / run: see ../ch6_1_foundations/01_first_chat.py for setup.
"""
from __future__ import annotations

import os
import sys

from genai_studio import GenAIStudio

CHAT_MODEL = "llama3.2:latest"


def estimate_tokens(text: str, method: str = "words") -> int:
    """Estimate token count from text.

    Word-based (1 word ~ 1.3 tokens) is decent for English prose; char-based
    (1 token ~ 4 chars) is better for mixed content (code, URLs, technical text).
    """
    if method == "words":
        return int(len(text.split()) * 1.3)
    if method == "chars":
        return len(text) // 4
    raise ValueError(f"Unknown method: {method}")


def check_fits_context(text: str, model_context: int = 8192,
                       system_tokens: int = 200, output_tokens: int = 1000) -> bool:
    """Check whether text fits within the context-window token budget."""
    estimated_input = estimate_tokens(text, "words")
    available = model_context - system_tokens - output_tokens
    fits = estimated_input <= available
    print(f"  context window:  {model_context:,} tokens")
    print(f"  system prompt:  -{system_tokens:,}")
    print(f"  output reserve: -{output_tokens:,}")
    print(f"  available:       {available:,} tokens")
    print(f"  input estimate:  {estimated_input:,} tokens")
    print(f"  fits: {'yes' if fits else 'NO - chunking required'}")
    return fits


def make_client() -> GenAIStudio:
    if not os.environ.get("GENAI_STUDIO_API_KEY"):
        sys.exit("GENAI_STUDIO_API_KEY is not set - see ../ch6_1_foundations/01_first_chat.py.")
    ai = GenAIStudio()
    ai.select_model(CHAT_MODEL)
    return ai


def main() -> None:
    # --- Heuristic estimates (no API needed) ---
    sample = "The bootstrap resamples data with replacement."
    print("Token estimates (heuristic):")
    print(f"  word-based: {estimate_tokens(sample, 'words')} tokens")
    print(f"  char-based: {estimate_tokens(sample, 'chars')} tokens")

    # --- Context-budget checks ---
    print("\nContext budget - short text:")
    check_fits_context("The mean is sensitive to outliers. " * 10)
    print("\nContext budget - long text:")
    check_fits_context("Statistical analysis reveals patterns. " * 5000)

    # --- Measured token usage from chat_complete() metadata (Exercise 6.3.1) ---
    ai = make_client()

    # The API reports token counts on every response. We need only the *prompt*
    # count, so we ask for a trivial reply (the completion content is irrelevant
    # here) and subtract a fixed-overhead baseline to isolate each text's tokens.
    instr = "Reply with only the word: ok\n\nText to measure: "
    try:
        base = ai.chat_complete(instr).prompt_tokens
    except Exception as exc:
        sys.exit(f"Could not reach the model to calibrate token counts: {exc}")

    demo = ai.chat_complete(instr + "The bootstrap resamples data with replacement.")
    print(f"\nchat_complete() token metadata: prompt={demo.prompt_tokens}, "
          f"completion={demo.completion_tokens}, total={demo.total_tokens}")

    content = {
        "prose": "The bootstrap method estimates sampling variability by repeatedly "
                 "drawing samples with replacement from the data.",
        "code": "def bootstrap(data, n=1000):\n    return [np.mean("
                "np.random.choice(data, len(data))) for _ in range(n)]",
        "urls": "Visit https://genai.rcac.purdue.edu/api/v1 and "
                "https://docs.python.org/3/library/statistics.html",
        "math": "E[X] = sum(x_i * P(x_i)), Var(X) = E[X^2] - (E[X])^2",
        "spanish": "El metodo bootstrap estima la variabilidad muestral mediante "
                   "remuestreo con reemplazo de los datos.",
    }
    print(f"\nContent-type token efficiency ({CHAT_MODEL}, fixed overhead removed):")
    for kind, text in content.items():
        try:
            marginal = ai.chat_complete(instr + text).prompt_tokens - base
        except Exception:
            print(f"  {kind:8s}: [skipped - backend dropped the response; rerun to retry]")
            continue
        words = len(text.split())
        print(f"  {kind:8s}: {marginal / words:.2f} tokens/word "
              f"({marginal} content tokens, {words} words)")
    print("\n  -> prose is most token-efficient; code, URLs, math, and non-English")
    print("     text pack more tokens per word, so a word-count heuristic under-counts")
    print("     them. Always measure on representative samples.")


if __name__ == "__main__":
    main()
