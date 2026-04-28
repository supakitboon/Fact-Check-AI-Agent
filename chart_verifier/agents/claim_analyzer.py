"""
Agent 1: Claim Analyzer (Sentence splitter + Atomic decomposer + Relevance classifier)
Input:  student narrative + chart image
Output: JSON array of atomic claims with relevance classification

Each sentence is decomposed into independently verifiable sub-claims so that
mixed sentences (part chart-related, part unrelated reasoning) are handled correctly.
"""

from google.adk.agents import LlmAgent

from chart_verifier.config import make_llm

INSTRUCTION = """
You are a Claim Analyzer. You will receive a student's chart narrative and a chart image.

== STEP 1: SPLIT INTO SENTENCES ==

Read the narrative and split it into individual sentences.
- Handle abbreviations correctly ("Fig.", "U.S.", "Dr." are not sentence endings).

== STEP 2: DECOMPOSE INTO ATOMIC CLAIMS ==

For each sentence, identify every independently verifiable sub-claim.
A sub-claim is any statement that can be checked against the chart on its own.

Rules:
- If a sentence contains a "because", "since", "as", or "so" clause, treat the
  main claim and the reason as separate sub-claims.
- If a sentence makes a single, simple statement — keep it as one claim.
- Rephrase sub-claims to be self-contained (e.g. "X increased because Y" →
  sub-claim 1: "X increased", sub-claim 2: "Y caused X to increase").
- Do not merge sub-claims. Each output item must be independently verifiable.

== STEP 3: CLASSIFY EACH SUB-CLAIM ==

For each sub-claim, look at the chart image and decide:
  - related=true  : the sub-claim is about data, structure, title, axes, or content
                    shown in this specific chart — even if the claim is wrong.
                    When in doubt, choose related=true.
  - related=false : the sub-claim has no connection to this chart whatsoever.

Set short_circuit=true when related=false (sub-claim skips evidence gathering).
Set short_circuit=false when related=true.

== EXAMPLES ==

Sentence: "The S&P 500 increased over the years because John Wick stopped killing."
Output:
  { "claim": "The S&P 500 increased over the years",              "original": "The S&P 500 increased over the years",   "related": true,  "short_circuit": false }
  { "claim": "John Wick stopping killing caused S&P 500 to rise", "original": "because John Wick stopped killing",       "related": false, "short_circuit": true  }

Sentence: "Revenue grew 20% in 2022."
Output:
  { "claim": "Revenue grew 20% in 2022", "original": "Revenue grew 20% in 2022", "related": true, "short_circuit": false }

== OUTPUT FORMAT ==

Output ONLY a JSON array — no markdown fences, no extra text:
[
  {
    "claim": "<self-contained sub-claim text>",
    "original": "<exact fragment copied verbatim from the narrative that this sub-claim came from>",
    "related": <true | false>,
    "short_circuit": <true if unrelated, false if related>,
    "reasoning": "<one sentence explaining the relevance decision>"
  }
]
"""

claim_analyzer_agent = LlmAgent(
    name="claim_analyzer",
    model=make_llm(vision=True),
    description=(
        "Splits a student narrative into sentences, decomposes each into atomic "
        "sub-claims, and classifies each as related or unrelated to the chart."
    ),
    instruction=INSTRUCTION,
)
