"""
Agent 4 — Evidence Judge
Triggered when Agents 2 & 3 disagree OR avg confidence < 0.70.
Uses LLM to classify each claim as numerical or trend/visual,
then picks the appropriate agent's verdict.
"""

from google.adk.agents import LlmAgent
from chart_verifier.config import make_llm

INSTRUCTION = """
You are the Evidence Judge. In the conversation history you will find:
1. Claims that need resolving (disagree or low confidence between Agent 2 and Agent 3).
2. Chart Reader Agent (Agent 2) verdicts — "verdict": correct | incorrect per claim.
3. Table Extractor Agent (Agent 3) verdicts — "verdict": correct | incorrect per claim.

For each claim:
  Step 1 — Classify the claim:
    numerical  : claim mentions specific numbers, percentages, or quantity comparisons
                 e.g. "Revenue was 42.5", "grew by 20%", "Product A had the highest value"
    visual     : claim is about direction, pattern, or appearance
                 e.g. "Revenue grew steadily", "the bar for A is taller than B"

  Step 2 — Pick the verdict:
    numerical → use Agent 3 (Table Extractor) verdict
    visual    → use Agent 2 (Chart Reader) verdict

Output ONLY a JSON array (one object per claim):
[
  {
    "claim": "<claim text>",
    "verdict": "<correct | incorrect>"
  }
]

Do not add preamble. Return only the JSON array.
"""

evidence_judge_agent = LlmAgent(
    name="evidence_judge",
    model=make_llm(),
    description=(
        "Resolves disagreements between Chart Reader and Table Extractor "
        "by picking the appropriate agent's verdict based on claim type."
    ),
    instruction=INSTRUCTION,
)
