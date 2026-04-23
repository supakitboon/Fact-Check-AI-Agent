"""
CLI entry point for the Chart Verifier pipeline.

Builds a multimodal message (narrative text + chart image) and runs it through
the root_agent SequentialAgent defined in agent.py.

Usage:
    python -m chart_verifier.orchestrator "<narrative>" "<image_path>"
"""

import asyncio
import json as _json
import os
from dotenv import load_dotenv

load_dotenv()

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from chart_verifier.agents.visual_evidence import encode_image_to_base64

APP_NAME = "chart_verifier"
USER_ID = "student_user"
AGENT_TIMEOUT = 300  # seconds — full pipeline can take a while


_SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
_MAX_NARRATIVE_CHARS = 10_000


def _validate_inputs(narrative: str, image_path: str) -> None:
    if not narrative or not narrative.strip():
        raise ValueError("Narrative must not be empty.")
    if len(narrative) > _MAX_NARRATIVE_CHARS:
        raise ValueError(f"Narrative exceeds {_MAX_NARRATIVE_CHARS} character limit.")
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in _SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image format '{ext}'. Use: {', '.join(_SUPPORTED_EXTENSIONS)}")


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
                    data = _json.loads(text[start : i + 1])
                    return data if isinstance(data, list) else None
                except Exception:
                    return None
    return None


def _extract_json_object(text: str) -> dict | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    data = _json.loads(text[start:i + 1])
                    return data if isinstance(data, dict) else None
                except Exception:
                    return None
    return None


def _resolve_claim_verdicts(captured: dict[str, str]) -> list[dict]:
    """
    Parse agent outputs and return [{claim, verdict, note}] where
    verdict ∈ {supported, contradicted, unrelated}.
    Prefers the structured JSON output from verdict_feedback if available.
    """
    # Prefer feedback_writer (has feedback text), fall back to verdict_arbiter (verdicts only)
    for agent_key in ("feedback_writer", "verdict_arbiter", "verdict_feedback"):
        vf_text = captured.get(agent_key, "")
        if not vf_text:
            continue
        data = _extract_json_object(vf_text)
        if data and "claims" in data:
            valid = {"supported", "contradicted", "unrelated"}
            results = []
            for item in data.get("claims", []):
                if isinstance(item, dict) and item.get("claim") and item.get("verdict", "").lower() in valid:
                    results.append({
                        "claim": item["claim"],
                        "verdict": item["verdict"].lower(),
                        "note": item.get("feedback", ""),
                    })
            if results:
                return results

    # Fall back to evidence-based resolution
    result: list[dict] = []

    # Unrelated claims — already resolved
    arr = _extract_json_array(captured.get("unrelated_claims", ""))
    if arr:
        for item in arr:
            if isinstance(item, dict) and item.get("claim"):
                result.append({"claim": item["claim"], "verdict": "unrelated"})

    def _parse_evidence(text: str) -> dict[str, dict]:
        by_claim: dict[str, dict] = {}
        arr = _extract_json_array(text or "")
        if not arr:
            return by_claim
        for item in arr:
            if isinstance(item, dict) and item.get("claim"):
                by_claim[item["claim"]] = item
        return by_claim

    def _map(v: str) -> str:
        v = (v or "").lower().strip()
        if v == "correct":   return "supported"
        if v == "incorrect": return "contradicted"
        return "unknown"

    vis_by_claim  = _parse_evidence(captured.get("visual_evidence", ""))
    strc_by_claim = _parse_evidence(captured.get("structured_evidence", ""))

    for claim in set(list(vis_by_claim) + list(strc_by_claim)):
        vis  = vis_by_claim.get(claim)
        strc = strc_by_claim.get(claim)

        vis_v     = _map(vis.get("verdict")  if vis  else "")
        strc_v    = _map(strc.get("verdict") if strc else "")
        vis_conf  = float(vis.get("confidence")  or 0.0) if vis  else 0.0
        strc_conf = float(strc.get("confidence") or 0.0) if strc else 0.0

        # Confidence: supported → +conf, contradicted → −conf
        def _signed(v, c):
            if v == "supported":    return +c
            if v == "contradicted": return -c
            return None

        scores = [s for s in [_signed(vis_v, vis_conf), _signed(strc_v, strc_conf)] if s is not None]
        if scores:
            avg_conf = sum(scores) / len(scores)
            if abs(avg_conf) >= 0.5:
                final = "supported" if avg_conf > 0 else "contradicted"
            else:
                # Below threshold — prefer structured for numerical, visual otherwise
                if strc_v != "unknown":
                    final = strc_v
                elif vis_v != "unknown":
                    final = vis_v
                else:
                    final = "unknown"
        else:
            final = "unknown"

        result.append({"claim": claim, "verdict": final})

    return result


async def run_pipeline(narrative: str, image_path: str) -> tuple[str, list[dict]]:
    """
    Run the full chart fact-checking pipeline.

    Args:
        narrative:  Student's written narrative paragraph.
        image_path: Path to the chart image file (PNG/JPG).

    Returns:
        (feedback_text, claim_verdicts) where claim_verdicts is a list of
        {claim, verdict} dicts with verdict ∈ {supported, contradicted, unrelated}.
    """
    _validate_inputs(narrative, image_path)

    # Import here to avoid circular imports (agent.py no longer imports orchestrator)
    from chart_verifier.agent import root_agent

    print("\n🔍 Starting Chart Verifier Pipeline...")
    print(f"   Narrative : {narrative[:80]}{'...' if len(narrative) > 80 else ''}")
    print(f"   Chart     : {image_path}\n")

    # Encode image once — sent as inline_data in the initial message
    print("🖼️  Encoding chart image...")
    image_b64 = encode_image_to_base64(image_path)

    # Detect MIME type from extension
    ext = os.path.splitext(image_path)[1].lower()
    mime_type = _MIME_TYPES[ext]

    # Build session and runner around the SequentialAgent pipeline
    session_id = "pipeline_session"
    svc = InMemorySessionService()
    await svc.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
    runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=svc)

    # Multimodal message: narrative text + chart image
    content = types.Content(
        role="user",
        parts=[
            types.Part(
                text=(
                    "Please fact-check the following student narrative against "
                    "the chart image provided.\n\n"
                    f"Student narrative:\n{narrative}"
                )
            ),
            types.Part(
                inline_data=types.Blob(
                    mime_type=mime_type,
                    data=image_b64,
                )
            ),
        ],
    )

    print("⚡ Running pipeline...\n")

    async def _run():
        last_text = ""
        captured: dict[str, str] = {}
        async for event in runner.run_async(
            user_id=USER_ID, session_id=session_id, new_message=content
        ):
            if event.author and event.content and event.content.parts:
                for part in event.content.parts:
                    text = getattr(part, "text", None)
                    if text and text.strip():
                        captured[event.author] = text
            if event.is_final_response() and event.content and event.content.parts:
                last_text = event.content.parts[0].text
        return last_text, captured

    last_text, captured = await asyncio.wait_for(_run(), timeout=AGENT_TIMEOUT)
    claim_verdicts = _resolve_claim_verdicts(captured)

    # Extract summary — feedback_writer has it, verdict_arbiter does not
    fw_data = _extract_json_object(captured.get("feedback_writer", ""))
    feedback_text = (fw_data or {}).get("summary", "") or last_text

    print("\n" + "=" * 60)
    print("STUDENT FEEDBACK")
    print("=" * 60)
    print(feedback_text)
    print("=" * 60)

    return feedback_text, claim_verdicts


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m chart_verifier.orchestrator <narrative> <image_path>")
        print("\nExample:")
        print(
            '  python -m chart_verifier.orchestrator '
            '"Revenue increased from 2020 to 2022." chart.png'
        )
        sys.exit(1)

    asyncio.run(run_pipeline(sys.argv[1], sys.argv[2]))
