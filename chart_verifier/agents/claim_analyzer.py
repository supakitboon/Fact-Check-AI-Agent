"""
Agent 1+2: Claim Analyzer (Decomposer + Relevance combined)
Input:  student narrative + chart image
Output: JSON array of atomic claims with relevance classification

Combines claim decomposition and relevance typing into a single pass,
saving one LLM round-trip and avoiding the intermediate JSON handoff.
"""

from google.adk.agents import LlmAgent

from chart_verifier.config import make_llm
from chart_verifier.tools.text_tools import split_sentences

INSTRUCTION = """
You are a Claim Analyzer. You will receive a student's chart narrative and a chart image.

Your job is to:
1. Call the `split_sentences` tool on the narrative to get an initial sentence split.
2. For each sentence, decompose into atomic claims only when the sentence contains
   multiple facts that can each be verified independently against the chart — meaning
   one part could be true while another is false.
   If all parts of a sentence must be true or false together (e.g. a value with its
   timeframe, label, or unit), keep them as one claim. Do not extract sub-components
   of a single fact as separate claims.
3. Remove subjective opinions or vague statements — keep only things that could be
   true or false based on chart data.
4. Preserve original numbers and labels from the student's text exactly.
5. For each atomic claim, look at the chart image and decide:
     - related=true  : the claim is about data, structure, title, axes, or content
                       shown in this specific chart — even if the claim is wrong.
                       When in doubt, choose related=true.
     - related=false : the claim has no connection to this chart whatsoever.
6. Set short_circuit=true when related=false (claim will skip evidence gathering).
   Set short_circuit=false when related=true.

Output ONLY a JSON array (one object per atomic claim):
[
  {
    "claim": "<atomic claim text>",
    "related": <true | false>,
    "short_circuit": <true if unrelated, false if related>,
    "reasoning": "<one sentence explaining the relevance decision>"
  }
]

Do not add preamble or explanation. Return only the JSON array.
"""

claim_analyzer_agent = LlmAgent(
    name="claim_analyzer",
    model=make_llm(vision=True),
    description=(
        "Breaks a student chart narrative into atomic claims and classifies "
        "each one as related or unrelated to the chart in a single pass."
    ),
    instruction=INSTRUCTION,
    tools=[split_sentences],
)
