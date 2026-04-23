"""
Agent 2 — Visual Evidence Agent
Reads the chart image visually and evaluates each related claim.
"""

import base64
from google.adk.agents import LlmAgent

from chart_verifier.config import make_llm

INSTRUCTION = """
You are a Chart Reader Agent. In the conversation history you will find:
1. The chart image in the user's original message.
2. A JSON array of claims from the Claim Filter (the most recent JSON array).

For each claim, examine the chart image visually and decide:
  correct   — the chart supports the claim
  incorrect — the chart does not support the claim

"unrelated" is NOT a valid verdict. You MUST choose correct or incorrect.

Confidence scoring rules:
  0.9-1.0 — labels, bars, lines, or colors clearly and directly confirm or deny the claim
  0.6-0.8 — claim is mostly verifiable visually but requires some estimation or interpretation
  0.3-0.5 — chart is ambiguous, crowded, or hard to read for this specific claim
  0.0-0.2 — chart does not contain enough visual information to evaluate this claim

Output ONLY a JSON array (one object per claim, same order as input):
[
  {
    "claim": "<claim text>",
    "verdict": "<correct | incorrect>",
    "confidence": <float 0.0-1.0>,
    "reasoning": "<1-2 sentences>"
  }
]

Do not add preamble. Return only the JSON array.
"""

visual_evidence_agent = LlmAgent(
    name="visual_evidence",
    model=make_llm(vision=True),
    description=(
        "Reads the chart image visually and produces a verdict and confidence "
        "score for each related claim."
    ),
    instruction=INSTRUCTION,
)


def encode_image_to_base64(image_path: str) -> str:
    """Helper to encode a local image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
