"""
Agent 1: Claim Analyzer (Sentence splitter + Relevance classifier)
Input:  student narrative + chart image
Output: JSON array of sentences with relevance classification

Each sentence from the narrative is kept intact — no further decomposition.
"""

from google.adk.agents import LlmAgent

from chart_verifier.config import make_llm
from chart_verifier.tools.text_tools import split_sentences

INSTRUCTION = """
You are a Claim Analyzer. You will receive a student's chart narrative and a chart image.

Your job is to:
1. Call the `split_sentences` tool on the narrative to get the sentences.
2. Use each sentence exactly as returned — do not split, merge, or rephrase.
3. For each sentence, look at the chart image and decide:
     - related=true  : the sentence is about data, structure, title, axes, or content
                       shown in this specific chart — even if the claim is wrong.
                       When in doubt, choose related=true.
     - related=false : the sentence has no connection to this chart whatsoever.
4. Set short_circuit=true when related=false (sentence will skip evidence gathering).
   Set short_circuit=false when related=true.

Output ONLY a JSON array (one object per sentence):
[
  {
    "claim": "<original sentence text, unchanged>",
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
        "Splits a student chart narrative into sentences and classifies "
        "each one as related or unrelated to the chart."
    ),
    instruction=INSTRUCTION,
    tools=[split_sentences],
)
