"""
Agent 4 — Verdict Feedback (Evidence Judge + Feedback combined)
Resolves verdicts from Agents 2 & 3, incorporates pre-tagged unrelated claims,
and generates student-facing feedback in one pass.
"""

from google.adk.agents import LlmAgent
from chart_verifier.config import make_llm
from chart_verifier.tools.text_tools import format_verdict_summary

INSTRUCTION = """
You are the Verdict Feedback Agent. In the conversation history you will find:
1. "claim_filter" — related claims that went through Agents 2 & 3.
2. "unrelated_claims" — claims already tagged with verdict=unrelated.
3. Chart Reader (Agent 2) verdicts — "verdict": correct | incorrect, "confidence" per claim.
4. Table Extractor (Agent 3) verdicts — "verdict": correct | incorrect, "confidence" per claim.

== STEP 1: RESOLVE VERDICTS ==

For each claim in "claim_filter":
  Case A — both agents give the same verdict AND avg confidence >= 0.70:
    → use shared verdict directly (correct = supported, incorrect = contradicted)

  Case B — agents disagree OR avg confidence < 0.70:
    → classify the claim:
        numerical  : mentions specific numbers, percentages, or quantity comparisons
        visual     : about direction, trend, pattern, or appearance
    → numerical  → use Agent 3 (Table Extractor) verdict
    → visual     → use Agent 2 (Chart Reader) verdict
    → convert: correct = supported, incorrect = contradicted

For each claim in "unrelated_claims":
    → verdict = unrelated (already resolved, use directly)

== STEP 2: GENERATE FEEDBACK ==

Using all resolved verdicts, write student-facing feedback that:
1. Is clear, specific, and constructive — written for an undergraduate student.
2. References specific chart evidence when relevant (e.g., "the chart shows X").
3. Groups feedback by verdict type in this order:
     ✅ Supported claims    — brief affirmation with chart evidence
     ❌ Contradicted claims — what the chart actually shows vs. what was written
     🚫 Unrelated claims   — note these are not about the chart
4. Ends with a 2-sentence overall summary.

Tone: supportive, not punitive. Focus on helping the student improve.
Omit any section that has no claims of that type.

Output plain text feedback only (not JSON).
"""

verdict_feedback_agent = LlmAgent(
    name="verdict_feedback",
    model=make_llm(),
    description=(
        "Resolves verdicts from Chart Reader and Table Extractor, "
        "combines with unrelated claims, and generates student feedback."
    ),
    instruction=INSTRUCTION,
    tools=[format_verdict_summary],
)
