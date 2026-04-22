"""
Agent 4: Structured Evidence Agent
Input:  atomic claim + chart image (to extract a pseudo-table from)
Output: extracted data table, reasoned answer, confidence score

Motivation from literature: structured intermediate representations improve
interpretability and verifiability compared to direct visual QA alone.
This agent first extracts a pseudo-table from the chart, then reasons over it.
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
        "to evaluate all non-short-circuit claims."
    ),
    instruction=INSTRUCTION,
)
