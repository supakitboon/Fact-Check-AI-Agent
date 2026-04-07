# Chart Verifier — ADK Multi-Agent Pipeline

Fact-checks student chart narratives against chart images using Google ADK + OpenRouter.

## Project Structure

```
chart_verifier/
├── agent.py              # ADK root agent (entry point for adk web / adk run)
├── orchestrator.py       # Full programmatic pipeline with step-by-step logging
├── requirements.txt
├── .env.example          # Copy to .env and fill in your key
│
├── agents/
│   ├── claim_decomposer.py       # Agent 1: atomic claim extraction (LLM)
│   ├── claim_typing.py           # Agent 2: claim classification (LLM)
│   ├── visual_evidence.py        # Agent 3: chart image reading (vision LLM)
│   ├── structured_evidence.py    # Agent 4: table extraction + reasoning (vision LLM)
│   ├── verification_question.py  # Agent 5: targeted re-check (LLM, conditional)
│   ├── judge.py                  # Agent 6: arbitration (LLM)
│   └── feedback.py               # Agent 7: student feedback (LLM)
│
└── tools/
    └── text_tools.py     # Deterministic tools (no LLM): sentence split,
                          # confidence threshold gate, verdict aggregation
```

## What uses LLM vs. deterministic tools

| Component | Type | Reason |
|---|---|---|
| `split_sentences` | Tool (regex) | No reasoning needed |
| `check_confidence_threshold` | Tool (comparison) | Pure float comparison |
| `format_verdict_summary` | Tool (counter) | Pure aggregation |
| Agents 1–7 | LLM | Require semantic understanding or vision |

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY

# 3a. Run via ADK web UI
adk web  # from parent directory of chart_verifier/

# 3b. Run programmatically
python -m chart_verifier.orchestrator "Your narrative here." path/to/chart.png
```

## Choosing a model

Edit `.env` to swap models anytime:

```env
# Fast + cheap
OPENROUTER_MODEL=openrouter/google/gemini-2.0-flash-exp:free
OPENROUTER_VISION_MODEL=openrouter/google/gemini-2.0-flash-exp:free

# High quality
OPENROUTER_MODEL=openrouter/anthropic/claude-3.5-sonnet
OPENROUTER_VISION_MODEL=openrouter/anthropic/claude-3.5-sonnet

# Cost-effective with vision
OPENROUTER_MODEL=openrouter/openai/gpt-4o-mini
OPENROUTER_VISION_MODEL=openrouter/openai/gpt-4o-mini
```

## Pipeline Flow

```
narrative + chart image
        │
        ▼
[Agent 1] Claim Decomposer → atomic claims
        │
        ▼ (per claim)
[Agent 2] Claim Typing → claim type
        │
        ├─ unrelated / unsupported → short-circuit verdict
        │
        ▼
[Agent 3] Visual Evidence  ──┐
[Agent 4] Structured Evidence─┤→ [Agent 5 if low confidence] → [Agent 6] Judge
                              │
                              ▼
                       [Agent 7] Feedback → student output
```

## Confidence & Re-check Logic

- Agents 3 & 4 each return a confidence score (0–1).
- If average confidence < 0.70, Agent 5 (Verification Question) is triggered.
- Agent 5 generates a targeted question and answers it from the available evidence.
- Agent 6 (Judge) receives all results and makes the final call.
