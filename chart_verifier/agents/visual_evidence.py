"""
Agent 3: Visual Evidence Agent
Input:  atomic claim + base64-encoded chart image
Output: evidence summary, candidate answer, confidence score

Why LLM (vision): Only a multimodal model can read the chart image directly.
This agent acts as a "sub-agent tool" that the Judge (Agent 6) calls.
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
        "Reads the chart image and produces visual evidence, candidate verdict, "
        "and confidence score for all non-short-circuit claims."
    ),
    instruction=INSTRUCTION,
)


def encode_image_to_base64(image_path: str) -> str:
    """Helper to encode a local image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
