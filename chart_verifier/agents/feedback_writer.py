"""
Agent 5 — Feedback Writer
Receives resolved verdicts from Agent 4 and writes student-facing feedback.
"""

from google.adk.agents import LlmAgent
from chart_verifier.config import make_llm

INSTRUCTION = """
You are the Feedback Writer. In the conversation history you will find:
1. "verdict_arbiter" — resolved verdicts from the Opus arbiter (present only when tiebreaking was needed).
2. "visual_evidence" and "structured_evidence" — raw Agent 2 & 3 verdicts (use these when verdict_arbiter is absent).
3. "unrelated_claims" — claims already tagged as unrelated.

To get the final verdict per claim:
  - If "verdict_arbiter" is present, use those verdicts directly.
  - Otherwise, compute avg confidence from Agents 2 & 3:
      correct/supported → +confidence, incorrect/contradicted → -confidence
      avg confidence = average; avg confidence > 0 → supported, < 0 → contradicted

Each claim has a final verdict: supported | contradicted | unrelated.

== YOUR TASK ==

For each claim write 1-2 sentences of student-facing feedback:
  - supported    : affirm and cite the specific chart evidence that confirms it.
  - contradicted : state what the chart actually shows vs what the student wrote.
  - unrelated    : note that this cannot be verified from the chart provided.

Then write a 2-3 sentence overall summary that is supportive, not punitive.
Focus on what the student did well and where to improve.

== OUTPUT FORMAT ==

Output ONLY valid JSON — no markdown fences, no extra text:
{
  "claims": [
    {
      "claim": "<exact claim text, unchanged>",
      "verdict": "<supported | contradicted | unrelated>",
      "feedback": "<1-2 sentence explanation>"
    }
  ],
  "summary": "<2-3 sentence overall feedback>"
}
"""

feedback_writer_agent = LlmAgent(
    name="feedback_writer",
    model=make_llm(),
    description="Writes per-claim feedback and overall summary from resolved verdicts.",
    instruction=INSTRUCTION,
)
