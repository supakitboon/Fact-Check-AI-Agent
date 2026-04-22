"""
ADK entry point — required for `adk web` and `adk run` commands.

Architecture:
  root_agent  ConditionalPipelineAgent: chart_verifier_orchestrator
    ├── Router              decides factcheck vs general — skips pipeline if general
    │
    ├── Agent 1: claim_analyzer     (decompose + relevance in one pass)
    │
    │   ── if ANY claim needs evidence ──────────────────────────────────────
    ├── evidence_gathering          ParallelAgent
    │     ├── Agent 2: visual_evidence
    │     └── Agent 3: structured_evidence
    │   ────────────────────────────────────────────────────────────────────
    │
    └── Agent 4: verdict_feedback   (judge + feedback combined)

If input is a general message, only the Router runs.
If ALL claims are short-circuit, evidence_gathering is skipped.
"""

import json
import logging
from contextlib import aclosing
from typing import AsyncGenerator

from google.adk.agents import BaseAgent, ParallelAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types

from chart_verifier.agents.general_chat import general_chat_agent
from chart_verifier.agents.claim_analyzer import claim_analyzer_agent
from chart_verifier.agents.visual_evidence import visual_evidence_agent
from chart_verifier.agents.structured_evidence import structured_evidence_agent
from chart_verifier.agents.verdict_feedback import verdict_feedback_agent


# ── Agents 2 & 3 run in parallel ──────────────────────────────────────────────
evidence_gathering = ParallelAgent(
    name="evidence_gathering",
    description="Runs Chart Reader and Table Extractor in parallel for all related claims.",
    sub_agents=[visual_evidence_agent, structured_evidence_agent],
)


class ConditionalPipelineAgent(BaseAgent):
    """Orchestrates the fact-checking pipeline with short-circuit routing."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        general_chat, analyzer, evidence, feedback = self.sub_agents

        # ── Step 0: route (pure Python — no LLM) ─────────────────────────
        if not _is_factcheck_request(ctx):
            async with aclosing(general_chat.run_async(ctx)) as gen:
                async for event in gen:
                    yield event
            return

        # ── Step 1: decompose + classify claims ───────────────────────────
        async with aclosing(analyzer.run_async(ctx)) as gen:
            async for event in gen:
                yield event

        # ── Step 2: split claims into related and unrelated ──────────────
        filtered, unrelated = _split_claims(ctx)
        yield Event(
            author="claim_filter",
            content=types.Content(
                role="model",
                parts=[types.Part(text=json.dumps(filtered, indent=2))],
            ),
        )
        yield Event(
            author="unrelated_claims",
            content=types.Content(
                role="model",
                parts=[types.Part(text=json.dumps(unrelated, indent=2))],
            ),
        )

        # ── Routing decision ──────────────────────────────────────────────
        if not self._all_claims_short_circuit(ctx):
            # ── Step 3: evidence gathering (parallel) ─────────────────────
            async with aclosing(evidence.run_async(ctx)) as gen:
                async for event in gen:
                    yield event

        # ── Step 4: judge + feedback (always runs) ────────────────────────
        async with aclosing(feedback.run_async(ctx)) as gen:
            async for event in gen:
                yield event

    def _all_claims_short_circuit(self, ctx: InvocationContext) -> bool:
        for event in reversed(ctx.session.events):
            if event.author != "claim_analyzer":
                continue
            if not (event.content and event.content.parts):
                continue
            for part in event.content.parts:
                text = getattr(part, "text", None)
                if not text:
                    continue
                claims = _extract_json_array(text)
                if claims is not None:
                    return all(c.get("short_circuit", False) for c in claims)
        return False


def _is_factcheck_request(ctx: InvocationContext) -> bool:
    """Returns True if the user's message contains both an image and text."""
    for event in ctx.session.events:
        if event.content and event.content.role == "user":
            parts = event.content.parts or []
            has_image = any(getattr(p, "inline_data", None) is not None for p in parts)
            has_text = any(getattr(p, "text", None) for p in parts)
            if has_image and has_text:
                return True
    return False


def _split_claims(ctx: InvocationContext) -> tuple[list, list]:
    """
    Splits claim_analyzer output into:
    - filtered: related claims (short_circuit=false) for Agents 2 & 3
    - unrelated: short_circuit=true claims pre-tagged with verdict=unrelated
    """
    for event in reversed(ctx.session.events):
        if event.author != "claim_analyzer":
            continue
        if not (event.content and event.content.parts):
            continue
        for part in event.content.parts:
            text = getattr(part, "text", None)
            if not text:
                continue
            claims = _extract_json_array(text)
            if claims is not None:
                filtered = [c for c in claims if not c.get("short_circuit", False)]
                unrelated = [
                    {"claim": c["claim"], "verdict": "unrelated"}
                    for c in claims if c.get("short_circuit", False)
                ]
                return filtered, unrelated
    return [], []


def _get_events_from(ctx: InvocationContext, author: str) -> list:
    """Return parsed JSON array from the most recent event by the given author."""
    for event in reversed(ctx.session.events):
        if event.author != author:
            continue
        if not (event.content and event.content.parts):
            continue
        for part in event.content.parts:
            text = getattr(part, "text", None)
            if not text:
                continue
            data = _extract_json_array(text)
            if data is not None:
                return data
    return []



def _extract_json_array(text: str) -> list | None:
    start = text.find("[")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                try:
                    data = json.loads(text[start : i + 1])
                    if isinstance(data, list) and data:
                        return data
                except json.JSONDecodeError as exc:
                    logging.warning("verdict_resolver JSON parse failed: %s", exc)
                    return None
    return None


# ── Root agent ────────────────────────────────────────────────────────────────
root_agent = ConditionalPipelineAgent(
    name="chart_verifier_orchestrator",
    description=(
        "Fact-checks a student chart narrative against a chart image. "
        "Skips evidence gathering when all claims are unrelated."
    ),
    sub_agents=[
        general_chat_agent,      # General conversation
        claim_analyzer_agent,    # Agent 1
        evidence_gathering,      # Agents 2 & 3 (parallel)
        verdict_feedback_agent,  # Agent 4 (judge + feedback combined)
    ],
)
