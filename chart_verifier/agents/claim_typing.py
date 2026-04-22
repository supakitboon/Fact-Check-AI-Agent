"""
Agent 2: Claim Relevance Agent
Input:  atomic claim + chart image
Output: JSON array — one entry per claim with related/unrelated decision

Simplified design: Agent 2 only decides whether the claim is about this specific
chart. If unrelated, the pipeline short-circuits immediately. All other claims
proceed to Agents 3 & 4 for evidence gathering.
"""

from google.adk.agents import LlmAgent

from chart_verifier.config import make_llm

INSTRUCTION = """
You are a Claim Relevance Agent. You will receive a chart image and a JSON array
of atomic claims.

For EACH claim, look at the chart image and decide:
  - "related"   : the claim is about data, structure, title, axes, or content
                  shown in this specific chart — even if the claim is wrong
  - "unrelated" : the claim has no connection to this chart whatsoever

A claim is "related" if the chart could in principle be used to verify it.
When in doubt, choose "related" — it is better to check than to skip.

Output ONLY a JSON array (one object per claim, same order):
[
  {
    "claim": "<original claim text>",
    "related": <true | false>,
    "reasoning": "<one sentence explaining your decision>"
  }
]

Do not add any preamble. Return only the JSON array.
"""

claim_typing_agent = LlmAgent(
    name="claim_typing",
    model=make_llm(),
    description=(
        "Classifies all atomic claims into types and flags short-circuit claims "
        "(unrelated / unsupported_interpretation) that skip evidence gathering."
    ),
    instruction=INSTRUCTION,
)
