"""
General Chat Agent — handles non-fact-check messages.
Responds conversationally when the user is not submitting a chart to verify.
"""

from google.adk.agents import LlmAgent
from chart_verifier.config import make_llm

INSTRUCTION = """
You are a helpful assistant for a chart fact-checking system.
The user is not submitting a chart to fact-check right now — respond naturally to whatever they say.

If the user expresses intent to check a chart or narrative but has not uploaded an image yet,
tell them to upload the chart image together with their written narrative and you will verify it.

Keep responses concise and friendly.
"""

general_chat_agent = LlmAgent(
    name="general_chat",
    model=make_llm(),
    description="Handles general conversation when the user is not submitting a fact-check request.",
    instruction=INSTRUCTION,
)
