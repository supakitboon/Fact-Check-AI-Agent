"""
Agent 5: Verification Question Agent
Input:  conversation history containing typed claims + visual + structured evidence
Output: JSON array — one recheck entry per claim (skipped for short-circuit and
        high-confidence claims, active only when avg confidence < 0.70)

Design choice: generating a *specific* narrow question is more reliable
than re-running the full claim because it constrains the LLM to answer
something precise and answerable.
"""

import os
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/anthropic/claude-3.5-sonnet")

INSTRUCTION = """
You are a Verification Question Agent. In the conversation history you will find:
1. A JSON array of typed claims from the Claim Typing Agent.
2. Visual evidence results from the Visual Evidence Agent (a JSON array).
3. Structured evidence results from the Structured Evidence Agent (a JSON array).

For each claim (match by "claim" text across all arrays):
  - If "short_circuit" is true                        → set needed=false.
  - If avg(visual_confidence, structured_confidence) >= 0.70  → set needed=false.
  - Otherwise (low confidence or conflicting evidence) → set needed=true and run:
      Step 1: Generate ONE specific, narrow question answerable from a chart.
              Examples:
                "What is the exact value of Product A in 2022?"
                "Did Revenue increase or decrease from 2020 to 2022?"
      Step 2: Using the provided evidence summaries, answer your own question.

Output ONLY a JSON array (one object per claim, same order as typed claims):
[
  {
    "claim": "<claim text>",
    "needed": <true | false>,
    "verification_question": "<question or null>",
    "answer_from_evidence": "<answer or null>",
    "resolved_verdict": "<supported | contradicted | partially_supported | insufficient_evidence | unrelated | null>",
    "confidence": <float 0.0-1.0 | null>
  }
]

Do not add preamble. Return only the JSON array.
"""

verification_question_agent = LlmAgent(
    name="verification_question",
    model=LiteLlm(
        model=MODEL,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        api_base="https://openrouter.ai/api/v1",
    ),
    description=(
        "Conditionally generates targeted verification questions for low-confidence "
        "claims (avg confidence < 0.70); skips high-confidence and short-circuit claims."
    ),
    instruction=INSTRUCTION,
)
