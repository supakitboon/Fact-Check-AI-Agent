# Chart Verifier — ADK Multi-Agent Pipeline

Fact-checks student chart narratives against chart images using Google ADK + OpenRouter.

## Project Structure

```
chart_verifier/
├── agent.py              # ADK root agent (ConditionalPipelineAgent — entry for adk web / adk run)
├── orchestrator.py       # Programmatic CLI runner with step-by-step logging
├── config.py             # Model config (OpenRouter via LiteLlm)
├── requirements.txt
├── .env                  # OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENROUTER_VISION_MODEL
│
├── agents/
│   ├── general_chat.py         # General conversation handler (non-factcheck path)
│   ├── claim_analyzer.py       # Agent 1: atomic claim extraction + relevance classification (vision LLM)
│   ├── visual_evidence.py      # Agent 2: chart image reading → verdict + confidence (vision LLM)
│   ├── structured_evidence.py  # Agent 3: pseudo-table extraction → verdict + confidence (vision LLM)
│   └── verdict_feedback.py     # Agent 4: verdict resolution + student feedback (LLM)
│
└── tools/
    ├── text_tools.py     # Deterministic tools: split_sentences, check_confidence_threshold,
    │                     # format_verdict_summary
    └── code_executor.py  # Sandboxed Python subprocess runner
```

## Pipeline Flow

```
narrative + chart image
        │
        ▼
[Router] image + text present?
        │
        ├─ No  → general_chat (conversational response)
        │
        └─ Yes ─────────────────────────────────────────────────
                │
                ▼
        [Agent 1] Claim Analyzer
          • splits narrative into atomic claims
          • classifies each claim as related/unrelated to the chart
          • unrelated claims are short-circuited (verdict = unrelated)
                │
                ├─ all claims unrelated? → skip evidence gathering
                │
                ▼
        [Agent 2] Visual Evidence  ──┐  (parallel)
        [Agent 3] Structured Evidence┘
          • Agent 2: reads chart image visually → verdict + confidence
          • Agent 3: extracts pseudo-table → reasons over data → verdict + confidence
                │
                ▼
        [Agent 4] Verdict Feedback
          • resolves verdicts from Agents 2 & 3
          • merges unrelated claims
          • writes student-facing feedback
```

## Verdict Resolution (Agent 4)

| Condition | Resolution |
|---|---|
| Both agents agree **and** avg confidence ≥ 0.70 | Use shared verdict directly |
| Agents disagree **or** avg confidence < 0.70 — numerical claim | Defer to Agent 3 (Structured Evidence) |
| Agents disagree **or** avg confidence < 0.70 — visual claim | Defer to Agent 2 (Visual Evidence) |
| Claim was short-circuited | verdict = unrelated |

## LLM vs. Deterministic

| Component | Type | Reason |
|---|---|---|
| `split_sentences` | Tool (regex) | No reasoning needed |
| `check_confidence_threshold` | Tool (comparison) | Pure float comparison |
| `format_verdict_summary` | Tool (counter) | Pure aggregation |
| `execute_verification_code` | Tool (subprocess) | Sandboxed execution |
| Agent 1 (Claim Analyzer) | Vision LLM | Reads chart image + semantic decomposition |
| Agent 2 (Visual Evidence) | Vision LLM | Reads chart image directly |
| Agent 3 (Structured Evidence) | Vision LLM | Reads chart image to extract table |
| Agent 4 (Verdict Feedback) | LLM | Semantic reasoning over text evidence |
| General Chat | LLM | Conversational response |

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY

# 3a. Run via ADK web UI
adk web  # from parent directory of chart_verifier/

# 3b. Run via Streamlit UI
streamlit run streamlit_app.py  # from project root

# 3c. Run programmatically
python -m chart_verifier.orchestrator "Your narrative here." path/to/chart.png
```

## Choosing a Model

Edit `.env` to swap models. `OPENROUTER_VISION_MODEL` is used by Agents 1, 2, and 3.

```env
# Fast + cheap
OPENROUTER_MODEL=openrouter/google/gemini-2.0-flash-exp:free
OPENROUTER_VISION_MODEL=openrouter/google/gemini-2.0-flash-exp:free

# High quality
OPENROUTER_MODEL=openrouter/anthropic/claude-3.7-sonnet
OPENROUTER_VISION_MODEL=openrouter/anthropic/claude-3.7-sonnet

# Cost-effective with vision
OPENROUTER_MODEL=openrouter/openai/gpt-4o-mini
OPENROUTER_VISION_MODEL=openrouter/openai/gpt-4o-mini
```
