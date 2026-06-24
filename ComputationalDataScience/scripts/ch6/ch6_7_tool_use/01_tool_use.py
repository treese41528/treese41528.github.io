"""
STAT 418 · Chapter 6.7 — Tool Use
Script 01: Tool Use — declare a tool, let the model call it, ground the answer.

Tool use (function calling) lets a model offload what it does poorly — exact
computation, live data, access to your systems — to functions you provide. The
model only *requests* a call; your code runs it and returns the result, which the
model then uses to write a grounded answer. We declare a tool with ``@tool``, run
one request -> execute -> return cycle, and confirm the model skips the tool when
it isn't needed.

Tool-calling support varies by model; we use ``qwen2.5:72b``, which supports it
natively on GenAI Studio. (``gemma3:12b``, the course chat model, does not.)

Prereqs / run: see ../ch6_1_foundations/01_first_chat.py for setup.
"""
from __future__ import annotations

import json
import os
import sys

from genai_studio import GenAIStudio
from genai_studio.agents import tool

TOOL_MODEL = "qwen2.5:72b"  # supports native tool-calling on GenAI Studio


@tool
def query_dataset(table: str, column: str, condition: str = "") -> str:
    """Return values from a named dataset, optionally filtered.

    Args:
        table: Name of the dataset to query (e.g. "patients").
        column: The column whose values to return.
        condition: Optional row filter, e.g. "age > 50".
    """
    # A real tool would query a database or DataFrame; we stub it with fixed data.
    data = {("patients", "cholesterol"): "172, 168, 195, 181, 204"}
    return data.get((table, column), "no rows")


# The tools we are willing to run. The dispatcher only executes a name found
# here — never a tool the model invents, and never with un-vetted arguments.
TOOLS = {"query_dataset": query_dataset}


def run_tool_call(ai: GenAIStudio, question: str) -> None:
    """One request -> execute -> return cycle, with a validating dispatcher."""
    messages = [{"role": "user", "content": question}]
    resp = ai.chat_raw(messages,
                       tools=[query_dataset.spec.to_openai()],
                       tool_choice="auto")
    msg = resp.choices[0].message

    if not getattr(msg, "tool_calls", None):
        # The model judged that no tool was needed and answered directly.
        print(f"  no tool call — direct answer: {msg.content!r}")
        return

    call = msg.tool_calls[0]
    print(f"  model requested: {call.function.name}({call.function.arguments})")

    # Safeguard: validate before executing — models can request an unknown tool or
    # malformed arguments, and you must never crash or blindly run them. How a
    # production system *responds* to a bad call is a design choice, not a fixed
    # rule: it might return the error to the model so it can retry (that feedback
    # loop is an agent — the SDK's Agent does exactly this via ToolRegistry.execute),
    # fall back to a safe default, cap the number of attempts, or escalate to a
    # human. Here we simply refuse, to show the validation step itself — the
    # mechanics, not the recovery policy.
    fn = TOOLS.get(call.function.name)
    if fn is None:
        print(f"  refused: {call.function.name!r} is not a registered tool "
              "(a real system would then choose how to respond — e.g. tell the "
              "model to retry, use a fallback, or stop)")
        return
    try:
        result = fn(**json.loads(call.function.arguments))
    except (json.JSONDecodeError, TypeError) as exc:
        print(f"  refused: invalid arguments ({exc})")
        return
    print(f"  tool result: {result}")

    # Send the result back WITHOUT tools, so the model writes prose rather than
    # calling the tool again (offered the tool, many models just re-call it).
    messages.append(msg)
    messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    final = ai.chat_raw(messages)
    print(f"  grounded answer: {final.choices[0].message.content}")


def make_client() -> GenAIStudio:
    if not os.environ.get("GENAI_STUDIO_API_KEY"):
        sys.exit("GENAI_STUDIO_API_KEY is not set — see ../ch6_1_foundations/01_first_chat.py.")
    ai = GenAIStudio()
    ai.select_model(TOOL_MODEL)
    return ai


def main() -> None:
    # 1. A @tool is an ordinary function plus a .spec — the schema the model sees.
    print("Tool schema the model receives (query_dataset.spec.parameters):")
    print(json.dumps(query_dataset.spec.parameters, indent=2))
    print("\ndirect call (no model):",
          query_dataset(table="patients", column="cholesterol", condition="age > 50"))

    ai = make_client()

    # 2. A question that needs the tool -> the model requests the call.
    print("\nQ1 (needs the tool) — average cholesterol of patients over 50:")
    run_tool_call(ai, "What is the average cholesterol of patients over 50?")

    # 3. A concept question the dataset can't answer. The model may reply directly
    #    or emit an unexpected call — the validating dispatcher handles both safely.
    print("\nQ2 (the dataset can't answer this) — a concept question:")
    run_tool_call(ai, "In one sentence, what is a p-value?")


if __name__ == "__main__":
    main()
