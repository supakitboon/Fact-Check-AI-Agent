"""
CLI entry point for the Chart Verifier pipeline.

Builds a multimodal message (narrative text + chart image) and runs it through
the root_agent SequentialAgent defined in agent.py.

Usage:
    python -m chart_verifier.orchestrator "<narrative>" "<image_path>"
"""

import asyncio
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


async def run_pipeline(narrative: str, image_path: str) -> str:
    """
    Run the full chart fact-checking pipeline.

    Args:
        narrative:  Student's written narrative paragraph.
        image_path: Path to the chart image file (PNG/JPG).

    Returns:
        Student-facing feedback string (from Agent 7 — Feedback).
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
        async for event in runner.run_async(
            user_id=USER_ID, session_id=session_id, new_message=content
        ):
            if event.is_final_response() and event.content and event.content.parts:
                last_text = event.content.parts[0].text
        return last_text

    feedback_text = await asyncio.wait_for(_run(), timeout=AGENT_TIMEOUT)

    print("\n" + "=" * 60)
    print("STUDENT FEEDBACK")
    print("=" * 60)
    print(feedback_text)
    print("=" * 60)

    return feedback_text


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
