"""
Agent 5: Verification Question Agent
Input:  conversation history containing typed claims + visual + structured evidence
Output: JSON array — one recheck entry per claim (skipped for short-circuit and
        high-confidence claims, active only when avg confidence < 0.70)

Design choice: generating a *specific* narrow question is more reliable
than re-running the full claim because it constrains the LLM to answer
something precise and answerable.
"""

from google.adk.agents import LlmAgent

from chart_verifier.config import make_llm

INSTRUCTION = """
You are a Verdict Resolver. In the conversation history you will find:
1. A filtered JSON array from the Claim Filter (related claims only — no unrelated claims).
2. Visual verdicts from the Chart Reader Agent (correct | incorrect + confidence).
3. Structured verdicts from the Table Extractor Agent (correct | incorrect + confidence).

For each claim from the Claim Filter, produce ONE resolved_verdict:

  Case A — both agents give the same verdict AND avg confidence >= 0.70:
    → correct   = resolved_verdict "supported"
    → incorrect = resolved_verdict "contradicted"
    Set needed=false.

  Case B — agents disagree OR avg confidence < 0.70:
    Set needed=true and resolve by:
      Step 1: Generate ONE specific, narrow question answerable from the chart.
              e.g. "Did revenue increase every year from 2020 to 2023?"
      Step 2: Answer your own question using the extracted table and visual evidence.
      Step 3: Assign resolved_verdict:
                supported    — the claim holds up against the chart
                contradicted — the claim does not hold up against the chart

Output ONLY a JSON array (one object per claim, same order as Claim Filter output):
[
  {
    "claim": "<claim text>",
    "needed": <true | false>,
    "verification_question": "<question or null if needed=false>",
    "answer_from_evidence": "<answer or null if needed=false>",
    "resolved_verdict": "<supported | contradicted — never null>",
    "confidence": <float 0.0-1.0>
  }
]

Do not add preamble. Return only the JSON array.
"""

verification_question_agent = LlmAgent(
    name="verification_question",
    model=make_llm(),
    description=(
        "Conditionally generates targeted verification questions for low-confidence "
        "claims (avg confidence < 0.70); skips high-confidence and short-circuit claims."
    ),
    instruction=INSTRUCTION,
)
