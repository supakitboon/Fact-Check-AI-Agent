"""
Router Agent — gates the pipeline.
Decides if the input is a fact-check request or a general message.
"""

from google.adk.agents import LlmAgent
from chart_verifier.config import make_llm

INSTRUCTION = """
You are a router for a chart fact-checking system.

Output "factcheck" if ANY of these conditions are true:
  1. There is a chart image attached AND there is any written text
  2. The user expresses intent to verify, check, or fact-check a chart or narrative
     e.g. "can you check my graph", "does my narrative match", "verify this chart",
          "i want to check my narrative", "is this correct based on the chart"

Output "general" ONLY if:
  - No image attached AND no fact-check intent expressed
  e.g. "hello", "what is S&P 500?", "how are you?"

Do NOT judge whether the narrative makes sense or is logical.
Do NOT evaluate content quality.

Output ONLY one word — either "factcheck" or "general". Nothing else.
"""

router_agent = LlmAgent(
    name="router",
    model=make_llm(vision=True),
    description="Decides if the input is a fact-check request or a general message.",
    instruction=INSTRUCTION,
)
