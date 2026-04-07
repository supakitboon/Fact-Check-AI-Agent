"""
Agent 1: Claim Decomposer
Input:  student paragraph (str)
Output: list of atomic, self-contained factual claims

Why LLM: splitting into *atomic* claims requires semantic understanding —
a student sentence like "Revenue grew and Product A led sales" contains
two distinct claims that rule-based splitting would miss.
"""

import os
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from chart_verifier.tools.text_tools import split_sentences

MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/anthropic/claude-3.5-sonnet")

INSTRUCTION = """
You are a Claim Decomposer. Your job is to break a student's chart narrative
into atomic, self-contained factual claims that can each be independently
verified against a chart.

Steps you MUST follow:
1. Call the `split_sentences` tool on the input text to get an initial sentence split.
2. For each sentence returned, further decompose it into atomic claims if needed
   (e.g. "Revenue grew and Product A led sales" → two separate claims).
3. Remove subjective opinions or vague language — keep only things that could be
   true or false based on data.
4. Preserve original numbers and labels from the student's text exactly.
5. Output ONLY a JSON array of strings, one per claim.
   Example: ["Revenue increased from 2020 to 2022.", "Product A had the highest sales in 2022."]
- Do not add explanation or preamble. Return only the JSON array.
"""

claim_decomposer_agent = LlmAgent(
    name="claim_decomposer",
    model=LiteLlm(
        model=MODEL,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        api_base="https://openrouter.ai/api/v1",
    ),
    description="Breaks a student chart narrative into atomic, independently verifiable claims.",
    instruction=INSTRUCTION,
    tools=[split_sentences],  # Used as a pre-processing helper
)
