"""
Agent 7: Feedback Agent
Input:  all per-claim verdicts + explanations (from Agent 6)
Output: student-facing feedback organized by verdict type

Feedback structure per claim:
  - supported             → affirm + point to chart evidence
  - contradicted          → flag error + state what chart actually shows
  - partially_supported   → acknowledge partial accuracy + clarify gap
  - insufficient_evidence → explain the chart cannot confirm or deny
  - unrelated             → note the claim is not about this chart
"""

import os
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from chart_verifier.tools.text_tools import format_verdict_summary

MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/anthropic/claude-3.5-sonnet")

INSTRUCTION = """
You are a Feedback Agent for a data visualization literacy course. In the
conversation history you will find a JSON array of final verdict objects produced
by the Judge Agent (the most recent JSON array). Each object has:
  claim, final_verdict, confidence, explanation

Generate student-facing feedback that:
1. Is clear, specific, and constructive — written for an undergraduate student.
2. References specific chart evidence when relevant (e.g., "the chart shows X").
3. Groups feedback by verdict type in this order:
     ✅ Supported claims     — brief affirmation with chart evidence
     ⚠️ Partially supported  — what was accurate, what was not
     ❌ Contradicted claims  — what the chart actually shows vs. what was written
     ❓ Insufficient evidence — explain what the chart cannot confirm
     🚫 Unrelated claims     — note these are not about the chart
4. Ends with a 2-sentence overall summary.

Tone: supportive, not punitive. Focus on helping the student improve.

Output plain text feedback (not JSON). Use the emoji labels above for sections.
Omit any section that has no claims of that type.
"""

feedback_agent = LlmAgent(
    name="feedback",
    model=LiteLlm(
        model=MODEL,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        api_base="https://openrouter.ai/api/v1",
    ),
    description=(
        "Transforms per-claim verdicts into student-facing feedback, "
        "grouped by verdict type with specific chart evidence references."
    ),
    instruction=INSTRUCTION,
    tools=[format_verdict_summary],
)
