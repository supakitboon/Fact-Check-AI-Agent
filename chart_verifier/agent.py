"""
ADK entry point — required for `adk web` and `adk run` commands.

Architecture:
  root_agent  ConditionalPipelineAgent: chart_verifier_orchestrator
    ├── Agent 1: claim_decomposer
    ├── Agent 2: claim_typing          (short-circuits unrelated / unsupported)
    │
    │   ── if ANY claim needs evidence ──────────────────────────────────────
    ├── evidence_gathering             ParallelAgent
    │     ├── Agent 3: visual_evidence
    │     └── Agent 4: structured_evidence
    ├── Agent 5: verification_question (conditional — skips high-confidence claims)
    │   ────────────────────────────────────────────────────────────────────
    │
    └── Agent 6: judge_feedback        (arbitration + student feedback in one pass)

If ALL claims are short-circuit (unrelated / unsupported_interpretation),
evidence_gathering and verification_question are skipped entirely.
"""

import json
import re
from contextlib import aclosing
from typing import AsyncGenerator, ClassVar

from google.adk.agents import BaseAgent, ParallelAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event

from chart_verifier.agents.claim_decomposer import claim_decomposer_agent
from chart_verifier.agents.claim_typing import claim_typing_agent
from chart_verifier.agents.visual_evidence import visual_evidence_agent
from chart_verifier.agents.structured_evidence import structured_evidence_agent
from chart_verifier.agents.verification_question import verification_question_agent
from chart_verifier.agents.judge_feedback import judge_feedback_agent


# ── Agents 3 & 4 run in parallel ──────────────────────────────────────────────
evidence_gathering = ParallelAgent(
    name="evidence_gathering",
    description=(
        "Runs visual evidence (Agent 3) and structured evidence (Agent 4) "
        "in parallel for all non-short-circuit claims."
    ),
    sub_agents=[visual_evidence_agent, structured_evidence_agent],
)


class ConditionalPipelineAgent(BaseAgent):
    """
    Orchestrates the fact-checking pipeline with short-circuit routing.

    After claim_typing, checks whether ALL claims are short-circuit
    (unrelated / unsupported_interpretation). If so, skips evidence_gathering
    and verification_question and jumps straight to judge_feedback.
    """

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        decomposer, typing, evidence, verification, judge = self.sub_agents

        # ── Step 1 & 2: always run ────────────────────────────────────────
        async with aclosing(decomposer.run_async(ctx)) as gen:
            async for event in gen:
                yield event

        async with aclosing(typing.run_async(ctx)) as gen:
            async for event in gen:
                yield event

        # ── Routing decision ──────────────────────────────────────────────
        if not self._all_claims_short_circuit(ctx):
            # ── Step 3 & 4: evidence gathering (parallel) ─────────────────
            async with aclosing(evidence.run_async(ctx)) as gen:
                async for event in gen:
                    yield event

            # ── Step 5: verification questions ───────────────────────────
            async with aclosing(verification.run_async(ctx)) as gen:
                async for event in gen:
                    yield event

        # ── Step 6: judge + feedback (always runs) ────────────────────────
        async with aclosing(judge.run_async(ctx)) as gen:
            async for event in gen:
                yield event

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _all_claims_short_circuit(self, ctx: InvocationContext) -> bool:
        """
        Returns True only when every typed claim has short_circuit=true.
        Scans backwards through session events for the claim_typing JSON output.
        """
        try:
            for event in reversed(ctx.session.events):
                if event.author != "claim_typing":
                    continue
                if not (event.content and event.content.parts):
                    continue
                for part in event.content.parts:
                    text = getattr(part, "text", None)
                    if not text:
                        continue
                    match = re.search(r"\[.*\]", text, re.DOTALL)
                    if not match:
                        continue
                    claims = json.loads(match.group())
                    if isinstance(claims, list) and claims:
                        result = all(c.get("short_circuit", False) for c in claims)
                        return result
        except Exception:
            pass
        # Default: do not skip evidence gathering
        return False


# ── Root agent ────────────────────────────────────────────────────────────────
root_agent = ConditionalPipelineAgent(
    name="chart_verifier_orchestrator",
    description=(
        "Fact-checks a student chart narrative against a chart image. "
        "Skips evidence gathering when all claims are unrelated or unsupported."
    ),
    sub_agents=[
        claim_decomposer_agent,       # Agent 1
        claim_typing_agent,           # Agent 2
        evidence_gathering,           # Agents 3 & 4 (parallel)
        verification_question_agent,  # Agent 5
        judge_feedback_agent,         # Agent 6
    ],
)
