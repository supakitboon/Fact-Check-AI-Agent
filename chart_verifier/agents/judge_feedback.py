"""
Agent 5: Feedback Agent
Input:  resolved verdicts from the Verification Agent (one label per claim)
Output: student-facing feedback text, grouped by verdict type
"""

from google.adk.agents import LlmAgent

from chart_verifier.config import make_llm
from chart_verifier.tools.text_tools import format_verdict_summary

INSTRUCTION = """
You are a Feedback Agent for a chart fact-checking pipeline.
In the conversation history you will find:
1. Resolved verdicts from "verdict_resolver" — each entry has "claim" and
   "resolved_verdict" (supported | contradicted).
2. All claims from "claim_analyzer" — use this to find claims where
   short_circuit=true and treat them as unrelated.

Use these directly — do not re-evaluate anything.

Generate student-facing feedback that:
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

judge_feedback_agent = LlmAgent(
    name="judge_feedback",
    model=make_llm(),
    description=(
        "Generates student-facing feedback from the resolved verdicts "
        "produced by the Verification Agent."
    ),
    instruction=INSTRUCTION,
    tools=[format_verdict_summary],
)
