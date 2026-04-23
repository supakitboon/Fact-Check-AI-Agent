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
    ├── Agent 4: verdict_arbiter    (Opus — resolve verdicts via avg confidence)
    │
    └── Agent 5: feedback_writer    (write student-facing feedback from verdicts)

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
from chart_verifier.agents.verdict_arbiter import verdict_arbiter_agent
from chart_verifier.agents.feedback_writer import feedback_writer_agent


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
        general_chat, analyzer, evidence, arbiter, feedback = self.sub_agents

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

        # ── Step 4: Opus arbiter — only if any claim needs tiebreaking ────
        # Compute avg confidence in Python; skip expensive Opus call when all
        # claims already have |avg confidence| >= TIEBREAK_THRESHOLD.
        if _needs_tiebreak(ctx):
            async with aclosing(arbiter.run_async(ctx)) as gen:
                async for event in gen:
                    yield event

        # ── Step 5: write student feedback ────────────────────────────────
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


TIEBREAK_THRESHOLD = 0.5


def _needs_tiebreak(ctx: InvocationContext) -> bool:
    """Return True if any claim has |avg confidence| < TIEBREAK_THRESHOLD.

    Reads visual_evidence and structured_evidence outputs, computes the avg
    confidence per claim, and escalates to Opus only when needed.
    """
    def _parse_evidence(author: str) -> dict[str, dict]:
        for event in reversed(ctx.session.events):
            if event.author != author:
                continue
            if not (event.content and event.content.parts):
                continue
            for part in event.content.parts:
                text = getattr(part, "text", None)
                if not text:
                    continue
                arr = _extract_json_array(text)
                if arr:
                    return {item["claim"]: item for item in arr if isinstance(item, dict) and item.get("claim")}
        return {}

    def _signed(verdict: str, confidence: float) -> float | None:
        v = (verdict or "").lower().strip()
        if v in ("correct", "supported"):
            return +confidence
        if v in ("incorrect", "contradicted"):
            return -confidence
        return None

    def _direction(v: str) -> str:
        if v in ("correct", "supported"):      return "supported"
        if v in ("incorrect", "contradicted"): return "contradicted"
        return "unknown"

    vis  = _parse_evidence("visual_evidence")
    strc = _parse_evidence("structured_evidence")

    all_claims = set(vis) | set(strc)
    if not all_claims:
        return False  # no evidence yet — skip arbiter

    for claim in all_claims:
        v_item = vis.get(claim, {})
        s_item = strc.get(claim, {})

        v2 = (v_item.get("verdict", "") or "").lower().strip()
        v3 = (s_item.get("verdict", "") or "").lower().strip()
        c2 = float(v_item.get("confidence") or 0.0)
        c3 = float(s_item.get("confidence") or 0.0)

        # Condition 1: agents disagree on verdict direction
        if v_item and s_item and _direction(v2) != _direction(v3) and _direction(v2) != "unknown" and _direction(v3) != "unknown":
            return True

        # Condition 2: avg confidence is too low
        scores = []
        for verdict, conf in ((v2, c2), (v3, c3)):
            s = _signed(verdict, conf)
            if s is not None:
                scores.append(s)
        if not scores:
            return True  # can't compute — escalate to be safe
        avg_conf = sum(scores) / len(scores)
        if abs(avg_conf) < TIEBREAK_THRESHOLD:
            return True

    return False  # all claims are clear-cut


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
        verdict_arbiter_agent,   # Agent 4 — verdict arbiter (Opus)
        feedback_writer_agent,   # Agent 5 — feedback writer
    ],
)
