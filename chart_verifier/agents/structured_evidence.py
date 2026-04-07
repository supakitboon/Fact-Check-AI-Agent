"""
Agent 4: Structured Evidence Agent
Input:  atomic claim + chart image (to extract a pseudo-table from)
Output: extracted data table, reasoned answer, confidence score

Motivation from literature: structured intermediate representations improve
interpretability and verifiability compared to direct visual QA alone.
This agent first extracts a pseudo-table from the chart, then reasons over it.
"""

import os
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

VISION_MODEL = os.getenv("OPENROUTER_VISION_MODEL", "openrouter/anthropic/claude-3.5-sonnet")

INSTRUCTION = """
You are a Structured Evidence Agent. In the conversation history you will find:
1. The chart image in the user's original message.
2. A JSON array of typed claims from the Claim Typing Agent (the most recent JSON array).

For each claim:
  - If "short_circuit" is true  → output a skipped entry (no chart reading needed).
  - If "short_circuit" is false → perform the two steps below.

Step 1 — Extract a data table:
  Read the chart and extract its data into a structured pseudo-table
  (e.g., rows of "Category | Year | Value"). Estimate from the chart scale if exact
  values are not readable.

Step 2 — Reason over the table:
  Using ONLY the extracted table (not visual intuition), evaluate the claim.

Output ONLY a JSON array (one object per claim, same order as input):
[
  {
    "claim": "<claim text>",
    "skipped": <true | false>,
    "extracted_table": "<markdown table — null if skipped>",
    "candidate_answer": "<supported | contradicted | partially_supported | insufficient_evidence | unrelated | null>",
    "confidence": <float 0.0-1.0 | null>,
    "reasoning": "<1-2 sentences showing how the table supports the verdict — null if skipped>"
  }
]

Do not add preamble. Return only the JSON array.
"""

structured_evidence_agent = LlmAgent(
    name="structured_evidence",
    model=LiteLlm(
        model=VISION_MODEL,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        api_base="https://openrouter.ai/api/v1",
    ),
    description=(
        "Extracts a structured data table from the chart image and reasons over it "
        "to evaluate all non-short-circuit claims."
    ),
    instruction=INSTRUCTION,
)
