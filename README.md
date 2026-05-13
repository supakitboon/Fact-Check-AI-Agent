# Chart Verifier — ADK Multi-Agent Pipeline

Fact-checks student chart narratives against chart images using Google ADK + OpenRouter.

## Project Structure

```
chart_verifier/
├── agent.py              # ADK root agent (ConditionalPipelineAgent — entry for adk web / adk run)
├── orchestrator.py       # Programmatic runner — resolves verdicts and attaches original fragments
├── config.py             # Model config (OpenRouter via LiteLlm)
├── requirements.txt
├── .env                  # OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENROUTER_VISION_MODEL, OPENROUTER_OPUS_MODEL
│
├── agents/
│   ├── general_chat.py         # General conversation handler (non-factcheck path)
│   ├── claim_analyzer.py       # Agent 1: sentence splitting + atomic decomposition + relevance classification (vision LLM)
│   ├── visual_evidence.py      # Agent 2: chart image reading → verdict + confidence (vision LLM)
│   ├── structured_evidence.py  # Agent 3: pseudo-table extraction → verdict + confidence (vision LLM)
│   ├── verdict_arbiter.py      # Agent 4: resolves conflicting/low-confidence verdicts (Opus)
│   └── feedback_writer.py      # Agent 5: writes per-claim feedback + overall summary (LLM)
│
└── tools/
    └── text_tools.py     # Utility: format_verdict_summary

evaluation/
├── eval_pipeline.py      # Ablation study across visual-only / structured-only / pipeline systems
└── eval_agent1.py        # Model ablation for Agent 1 across different vision LLMs

streamlit_app.py          # Streamlit UI — narrative highlighting + feedback display
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
        └─ Yes ─────────────────────────────────────────────────────────
                │
                ▼
        [Agent 1] Claim Analyzer  (vision LLM — single pass, no tools)
          • splits narrative into sentences
          • decomposes each sentence into atomic, independently verifiable sub-claims
            (e.g. "X rose because Y" → sub-claim 1: "X rose", sub-claim 2: "Y caused X to rise")
          • classifies each sub-claim as related or unrelated to the chart
          • stores "original" fragment for UI highlighting
          • unrelated sub-claims are short-circuited (verdict = unrelated, sent directly to Agent 5)
                │
                ├─ all claims unrelated? → skip to Agent 5
                │
                ▼
        [Agent 2] Visual Evidence  ──┐  (parallel)
        [Agent 3] Structured Evidence┘
          • Agent 2: reads chart image visually → verdict + confidence
          • Agent 3: extracts pseudo-table from image → verdict + confidence
                │
                ▼
        [Agent 4] Verdict Arbiter  (conditional — Opus)
          • runs ONLY when agents disagree OR avg confidence < 0.5
          • commits to a definitive supported / contradicted verdict
          • when skipped: Python resolves verdict from avg confidence
            and emits a verdict_resolved event for Agent 5 to read
                │
                ▼
        [Agent 5] Feedback Writer
          • reads unrelated sub-claims directly from unrelated_claims event (Agent 1 path)
          • reads related-claim verdicts from verdict_arbiter OR verdict_resolved (never recomputes)
          • writes 1-2 sentence per-claim feedback
          • writes 2-3 sentence overall student summary
```

## Verdict Resolution

| Condition | Resolution |
|---|---|
| Agents 2 & 3 agree **and** \|avg confidence\| ≥ 0.5 | Python resolves → emits `verdict_resolved` — Agent 4 skipped |
| Agents 2 & 3 disagree **or** \|avg confidence\| < 0.5 | Escalate to Agent 4 (Verdict Arbiter — Opus) |
| Sub-claim was short-circuited (unrelated) | verdict = unrelated, bypasses Agents 2, 3, and 4 entirely |

## Components

| Component | Type | Model |
|---|---|---|
| Agent 1 (Claim Analyzer) | Vision LLM | `OPENROUTER_VISION_MODEL` |
| Agent 2 (Visual Evidence) | Vision LLM | `OPENROUTER_VISION_MODEL` |
| Agent 3 (Structured Evidence) | Vision LLM | `OPENROUTER_VISION_MODEL` |
| Agent 4 (Verdict Arbiter) | LLM (Opus) | `OPENROUTER_OPUS_MODEL` |
| Agent 5 (Feedback Writer) | LLM | `OPENROUTER_MODEL` |
| General Chat | LLM | `OPENROUTER_MODEL` |
| Verdict resolution (Agent 4 skipped) | Python (deterministic) | — |

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

## Evaluation

```bash
# Ablation study across visual-only / structured-only / pipeline systems
python evaluation/eval_pipeline.py --n 20 --output eval_results.csv

# Agent 1 model ablation — compare vision LLMs on sentence splitting + relevance classification
python evaluation/eval_agent1.py --n 20 --output eval_agent1_results.csv

# Custom model list for Agent 1 ablation
python evaluation/eval_agent1.py --n 10 --models "openrouter/anthropic/claude-sonnet-4-6,openrouter/openai/gpt-4o"
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
