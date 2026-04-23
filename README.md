# Chart Verifier — ADK Multi-Agent Pipeline

Fact-checks student chart narratives against chart images using Google ADK + OpenRouter.

## Project Structure

```
chart_verifier/
├── agent.py              # ADK root agent (ConditionalPipelineAgent — entry for adk web / adk run)
├── orchestrator.py       # Programmatic CLI runner with step-by-step logging
├── config.py             # Model config (OpenRouter via LiteLlm)
├── requirements.txt
├── .env                  # OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENROUTER_VISION_MODEL, OPENROUTER_OPUS_MODEL
│
├── agents/
│   ├── general_chat.py         # General conversation handler (non-factcheck path)
│   ├── claim_analyzer.py       # Agent 1: sentence splitting + relevance classification (vision LLM)
│   ├── visual_evidence.py      # Agent 2: chart image reading → verdict + confidence (vision LLM)
│   ├── structured_evidence.py  # Agent 3: pseudo-table extraction → verdict + confidence (vision LLM)
│   ├── verdict_arbiter.py      # Agent 4: tiebreaker — resolves conflicting/low-confidence verdicts (Opus)
│   └── feedback_writer.py      # Agent 5: writes per-claim feedback + overall summary (LLM)
│
└── tools/
    └── text_tools.py     # Deterministic tools: split_sentences, check_confidence_threshold,
                          # format_verdict_summary
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
        └─ Yes ──────────────────────────────────────────────────────
                │
                ▼
        [Agent 1] Claim Analyzer
          • calls split_sentences tool on narrative
          • classifies each sentence as related/unrelated to the chart
          • unrelated sentences are short-circuited (verdict = unrelated)
                │
                ├─ all claims unrelated? → skip to Agent 5
                │
                ▼
        [Agent 2] Visual Evidence  ──┐  (parallel)
        [Agent 3] Structured Evidence┘
          • Agent 2: reads chart image visually → verdict + confidence
          • Agent 3: extracts pseudo-table from image → reasons over data → verdict + confidence
                │
                ▼
        [Agent 4] Verdict Arbiter  (conditional — Opus)
          • runs ONLY when agents disagree OR avg confidence < 0.5
          • acts as senior arbiter to commit to a definitive verdict
          • skipped when all claims already have clear-cut agreement
                │
                ▼
        [Agent 5] Feedback Writer
          • uses Agent 4 verdicts if available, else resolves from Agents 2 & 3 directly
          • writes 1-2 sentence per-claim feedback
          • writes 2-3 sentence overall student summary
```

## Verdict Resolution

| Condition | Resolution |
|---|---|
| Both agents agree **and** \|avg confidence\| ≥ 0.5 | Skip Agent 4 — use avg confidence verdict directly |
| Agents disagree **or** \|avg confidence\| < 0.5 | Escalate to Agent 4 (Verdict Arbiter — Opus) |
| Claim was short-circuited (unrelated) | verdict = unrelated, skip evidence gathering |

## LLM vs. Deterministic

| Component | Type | Model |
|---|---|---|
| `split_sentences` | Tool (regex) | — |
| `check_confidence_threshold` | Tool (comparison) | — |
| `format_verdict_summary` | Tool (counter) | — |
| Agent 1 (Claim Analyzer) | Vision LLM | `OPENROUTER_VISION_MODEL` |
| Agent 2 (Visual Evidence) | Vision LLM | `OPENROUTER_VISION_MODEL` |
| Agent 3 (Structured Evidence) | Vision LLM | `OPENROUTER_VISION_MODEL` |
| Agent 4 (Verdict Arbiter) | LLM (Opus) | `OPENROUTER_OPUS_MODEL` |
| Agent 5 (Feedback Writer) | LLM | `OPENROUTER_MODEL` |
| General Chat | LLM | `OPENROUTER_MODEL` |

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

Edit `.env` to swap models. `OPENROUTER_VISION_MODEL` is used by Agents 1, 2, and 3. `OPENROUTER_OPUS_MODEL` is used by Agent 4 (Verdict Arbiter) and defaults to `claude-opus-4-7`.

```env
# Fast + cheap
OPENROUTER_MODEL=openrouter/google/gemini-2.0-flash-exp:free
OPENROUTER_VISION_MODEL=openrouter/google/gemini-2.0-flash-exp:free

# High quality
OPENROUTER_MODEL=openrouter/anthropic/claude-3.7-sonnet
OPENROUTER_VISION_MODEL=openrouter/anthropic/claude-3.7-sonnet

# Verdict Arbiter (Agent 4) — defaults to Opus for tiebreaking quality
OPENROUTER_OPUS_MODEL=openrouter/anthropic/claude-opus-4-7
```