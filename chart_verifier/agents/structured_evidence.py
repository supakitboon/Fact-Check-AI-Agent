"""
Agent 3 — Structured Evidence Agent
Extracts a data table from the chart image and reasons over it to evaluate each claim.
"""

from google.adk.agents import LlmAgent

from chart_verifier.config import make_llm

INSTRUCTION = """
You are a Structured Evidence Agent. In the conversation history you will find:
1. The chart image in the user's original message.
2. A JSON array of claims from the Claim Filter (the most recent JSON array).

For each claim, perform the steps below:

Step 1 — Extract a data table:
  Read the chart and extract its data into a structured pseudo-table
  (e.g., rows of "Category | Year | Value"). Estimate from the chart scale if exact
  values are not readable.

Step 2 — Evaluate the claim against the table:
  Decide:
    correct   — the extracted data confirms the claim
    incorrect — the extracted data contradicts the claim

"unrelated" is NOT a valid verdict. You MUST choose correct or incorrect.

Confidence scoring rules:
  0.9-1.0 — exact values read directly from the chart confirm or deny the claim with no ambiguity
  0.6-0.8 — values estimated from scale are close enough to verify the claim with reasonable certainty
  0.3-0.5 — values are difficult to read precisely; significant estimation was needed
  0.0-0.2 — could not extract enough data from the chart to evaluate the claim reliably

Output ONLY a JSON array (one object per claim, same order as input):
[
  {
    "claim": "<claim text>",
    "extracted_table": "<markdown table>",
    "verdict": "<correct | incorrect>",
    "confidence": <float 0.0-1.0>,
    "reasoning": "<1-2 sentences>"
  }
]

Do not add preamble. Return only the JSON array.
"""

structured_evidence_agent = LlmAgent(
    name="structured_evidence",
    model=make_llm(vision=True),
    description=(
        "Extracts a structured data table from the chart image and reasons over it "
        "to evaluate each related claim."
    ),
    instruction=INSTRUCTION,
)
