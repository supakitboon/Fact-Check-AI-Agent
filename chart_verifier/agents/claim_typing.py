"""
Agent 2: Claim Typing Agent
Input:  JSON array of atomic claims from the Claim Decomposer (previous message)
Output: JSON array — one typed entry per claim, with short-circuit flag

Claim types:
  - numeric_lookup       : exact value from chart ("Revenue was $5M in 2022")
  - trend                : directional change over time ("Sales increased from 2020 to 2022")
  - comparison           : comparing two or more items ("A was higher than B")
  - ranking              : ordering of items ("A had the highest sales")
  - composition_share    : proportion or percentage ("A accounted for 40% of total")
  - unsupported_interpretation : conclusion beyond the data ("This shows success")
  - unrelated            : not about the chart at all

Why this matters:
  Each type routes to a different verification strategy downstream.
  'unrelated' and 'unsupported_interpretation' short-circuit — no evidence check needed.
"""

import os
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/anthropic/claude-3.5-sonnet")

INSTRUCTION = """
You are a Claim Typing Agent. In the conversation history you will find a JSON array
of atomic claims produced by the Claim Decomposer Agent.

For EACH claim in that array, classify it into exactly one of the following types:
  - numeric_lookup         : Refers to a specific value from the chart
  - trend                  : Describes direction of change over time
  - comparison             : Compares two or more categories or time points
  - ranking                : Describes ordering (highest, lowest, most, least)
  - composition_share      : Refers to a proportion, percentage, or share
  - unsupported_interpretation : A conclusion or interpretation beyond what data shows
  - unrelated              : Not related to the chart at all

Short-circuit rules:
  - "unrelated"                  → set short_circuit=true, short_circuit_verdict="unrelated"
  - "unsupported_interpretation" → set short_circuit=true, short_circuit_verdict="insufficient_evidence"
  - all others                   → set short_circuit=false, short_circuit_verdict=null

Output ONLY a JSON array (one object per claim, same order):
[
  {
    "claim": "<original claim text>",
    "type": "<claim type>",
    "reasoning": "<one sentence explaining your classification>",
    "short_circuit": <true | false>,
    "short_circuit_verdict": "<'unrelated' | 'insufficient_evidence' | null>"
  }
]

Do not add any preamble. Return only the JSON array.
"""

claim_typing_agent = LlmAgent(
    name="claim_typing",
    model=LiteLlm(
        model=MODEL,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        api_base="https://openrouter.ai/api/v1",
    ),
    description=(
        "Classifies all atomic claims into types and flags short-circuit claims "
        "(unrelated / unsupported_interpretation) that skip evidence gathering."
    ),
    instruction=INSTRUCTION,
)
