"""
Agent 6: Judge / Arbitration Agent
Input:  conversation history containing all evidence (Agents 3-5 outputs)
Output: JSON array of final verdicts — one per claim

Arbitration is claim-type-aware:
  - numeric_lookup  → trust structured evidence more (exact values)
  - trend/ranking   → balance both (visual gestalt + table direction)
  - composition     → trust structured more (percentages)
  - unrelated / unsupported_interpretation → use short_circuit_verdict directly
"""

import os
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from chart_verifier.tools.text_tools import check_confidence_threshold

MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/anthropic/claude-3.5-sonnet")

INSTRUCTION = """
You are the Judge Agent for a chart fact-checking pipeline. In the conversation
history you will find:
1. Typed claims from the Claim Typing Agent (JSON array with short_circuit flags).
2. Visual evidence from the Visual Evidence Agent (JSON array).
3. Structured evidence from the Structured Evidence Agent (JSON array).
4. Verification question results from the Verification Question Agent (JSON array).

For each claim (match by "claim" text):
  - If short_circuit=true → use the "short_circuit_verdict" directly as final_verdict.
  - Otherwise → apply the arbitration rules below.

Arbitration rules by claim type:
  - numeric_lookup:              Prefer structured evidence. Visual confirms.
  - trend:                       Balance both. Require directional agreement.
  - comparison:                  Balance both. Require agreement on which is higher/lower.
  - ranking:                     Balance both. Require agreement on ordering.
  - composition_share:           Prefer structured evidence (percentages).
  - unsupported_interpretation:  Always → insufficient_evidence.
  - unrelated:                   Always → unrelated.

If a verification question result has needed=true and high confidence, it takes priority
over visual/structured evidence for that claim.

Output ONLY a JSON array (one object per claim, same order as typed claims):
[
  {
    "claim": "<original claim>",
    "final_verdict": "<supported | contradicted | partially_supported | insufficient_evidence | unrelated>",
    "confidence": <float 0.0-1.0>,
    "explanation": "<2-3 sentences explaining the verdict with specific chart evidence>"
  }
]

Do not add preamble. Return only the JSON array.
"""

judge_agent = LlmAgent(
    name="judge",
    model=LiteLlm(
        model=MODEL,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        api_base="https://openrouter.ai/api/v1",
    ),
    description=(
        "Arbitrates between visual and structured evidence for all claims, "
        "applying claim-type-aware rules to produce final verdicts."
    ),
    instruction=INSTRUCTION,
    tools=[check_confidence_threshold],
)
