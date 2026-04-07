"""
Agent 3: Visual Evidence Agent
Input:  atomic claim + base64-encoded chart image
Output: evidence summary, candidate answer, confidence score

Why LLM (vision): Only a multimodal model can read the chart image directly.
This agent acts as a "sub-agent tool" that the Judge (Agent 6) calls.
"""

import os
import base64
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

VISION_MODEL = os.getenv("OPENROUTER_VISION_MODEL", "openrouter/anthropic/claude-3.5-sonnet")

INSTRUCTION = """
You are a Visual Evidence Agent. In the conversation history you will find:
1. The chart image in the user's original message.
2. A JSON array of typed claims from the Claim Typing Agent (the most recent JSON array).

For each claim:
  - If "short_circuit" is true  → output a skipped entry (no chart reading needed).
  - If "short_circuit" is false → examine the chart image and evaluate the claim visually.

Output ONLY a JSON array (one object per claim, same order as input):
[
  {
    "claim": "<claim text>",
    "skipped": <true | false>,
    "evidence_summary": "<what the chart visually shows — null if skipped>",
    "candidate_answer": "<supported | contradicted | partially_supported | insufficient_evidence | unrelated | null>",
    "confidence": <float 0.0-1.0 | null>,
    "reasoning": "<1-2 sentences — null if skipped>"
  }
]

Be precise. Use 'insufficient_evidence' when the chart lacks enough data to verify the claim.
Do not add preamble. Return only the JSON array.
"""

visual_evidence_agent = LlmAgent(
    name="visual_evidence",
    model=LiteLlm(
        model=VISION_MODEL,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        api_base="https://openrouter.ai/api/v1",
    ),
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
