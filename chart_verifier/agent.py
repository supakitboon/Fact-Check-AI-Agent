"""
ADK entry point — required for `adk web` and `adk run` commands.

Architecture:
  root_agent  SequentialAgent: chart_verifier_orchestrator
    ├── Agent 1: claim_decomposer
    ├── Agent 2: claim_typing        (short-circuits unrelated / unsupported)
    ├── evidence_gathering           ParallelAgent
    │     ├── Agent 3: visual_evidence
    │     └── Agent 4: structured_evidence
    ├── Agent 5: verification_question  (conditional — skips high-confidence claims)
    ├── Agent 6: judge
    └── Agent 7: feedback
"""

from google.adk.agents import SequentialAgent, ParallelAgent

from chart_verifier.agents.claim_decomposer import claim_decomposer_agent
from chart_verifier.agents.claim_typing import claim_typing_agent
from chart_verifier.agents.visual_evidence import visual_evidence_agent
from chart_verifier.agents.structured_evidence import structured_evidence_agent
from chart_verifier.agents.verification_question import verification_question_agent
from chart_verifier.agents.judge import judge_agent
from chart_verifier.agents.feedback import feedback_agent

# ── Agents 3 & 4 run in parallel ──────────────────────────────────────────────
evidence_gathering = ParallelAgent(
    name="evidence_gathering",
    description=(
        "Runs visual evidence (Agent 3) and structured evidence (Agent 4) "
        "in parallel for all non-short-circuit claims."
    ),
    sub_agents=[visual_evidence_agent, structured_evidence_agent],
)

# ── Full 7-agent pipeline ──────────────────────────────────────────────────────
root_agent = SequentialAgent(
    name="chart_verifier_orchestrator",
    description=(
        "Fact-checks a student chart narrative against a chart image "
        "using a 7-agent pipeline: decompose → type → evidence (parallel) "
        "→ verify → judge → feedback."
    ),
    sub_agents=[
        claim_decomposer_agent,        # Agent 1
        claim_typing_agent,             # Agent 2  (short-circuit logic inside)
        evidence_gathering,             # Agents 3 & 4 (parallel)
        verification_question_agent,    # Agent 5  (confidence check inside)
        judge_agent,                    # Agent 6
        feedback_agent,                 # Agent 7
    ],
)
