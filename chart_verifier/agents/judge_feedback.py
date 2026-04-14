"""
Agent 6+7: Judge & Feedback Agent (combined)
Input:  conversation history containing all evidence (Agents 3-5 outputs)
Output: student-facing feedback text, grouped by verdict type

This agent combines arbitration (Judge) and feedback generation (Feedback)
into a single pass — eliminating the intermediate JSON handoff.

Arbitration is claim-type-aware:
  - numeric_lookup  → trust structured evidence more (exact values)
  - trend/ranking   → balance both (visual gestalt + table direction)
  - composition     → trust structured more (percentages)
  - unrelated / unsupported_interpretation → use short_circuit_verdict directly
"""

import os
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from chart_verifier.tools.text_tools import check_confidence_threshold, format_verdict_summary

MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/anthropic/claude-3.5-sonnet")

INSTRUCTION = """
You are the Judge & Feedback Agent for a chart fact-checking pipeline.
In the conversation history you will find:
1. Typed claims from the Claim Typing Agent (JSON array with short_circuit flags).
2. Visual evidence from the Visual Evidence Agent (JSON array).
3. Structured evidence from the Structured Evidence Agent (JSON array).
4. Verification question results from the Verification Question Agent (JSON array).

== STEP 1: ARBITRATION ==

For each claim (match by "claim" text):
  - If short_circuit=true → use the "short_circuit_verdict" directly as final_verdict.
  - Otherwise → apply the arbitration rules below.

Arbitration rules by claim type:
  - numeric_lookup:              Prefer structured evidence. Visual confirms.
  - trend:                       Balance both. Require directional agreement.
  - comparison:                  Balance both. Require agreement on which is higher/lower.
  - ranking:                     Balance both. Require agreement on ordering.
  - composition_share:           Prefer structured evidence (percentages).
  - unsupported_interpretation:  Always → insufficient_evidence.
  - unrelated:                   Always → unrelated.

If a verification question result has needed=true and high confidence, it takes priority
over visual/structured evidence for that claim.

Verdict options: supported | contradicted | partially_supported | insufficient_evidence | unrelated

== STEP 2: STUDENT FEEDBACK ==

Using the verdicts from Step 1, generate student-facing feedback that:
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
Omit any section that has no claims of that type.

Output plain text feedback only (not JSON).
"""

judge_feedback_agent = LlmAgent(
    name="judge_feedback",
    model=LiteLlm(
        model=MODEL,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        api_base="https://openrouter.ai/api/v1",
    ),
    description=(
        "Arbitrates between visual and structured evidence for all claims, "
        "then generates student-facing feedback grouped by verdict type."
    ),
    instruction=INSTRUCTION,
    tools=[check_confidence_threshold, format_verdict_summary],
)
