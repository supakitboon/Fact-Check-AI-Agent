"""
All-in-one benchmark: tests every configured model against every agent,
saves one JSON per (agent, model) to output/, and prints a summary table.
Runs both classification metrics and ChartCheck-style NLG metrics (BLEU /
ROUGE-L / METEOR / BERTScore) for visual and structured agents automatically.

== CONFIGURE HERE ==============================================================

  Edit VISION_MODELS / TEXT_ONLY_MODELS before running.
  Set LIMIT to 20 for a quick sanity check, 100+ for reliable numbers.
  If a model fails on visual/structured with an image-related error, move it
  to TEXT_ONLY_MODELS so it only runs as the judge.

== RUN =========================================================================

  python run_benchmark.py

  Results: output/<agent>__<model-slug>.json + output/summary.json
"""

import asyncio
import json
import re
from pathlib import Path
from datetime import datetime

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / "chart_verifier" / ".env")

# ==============================================================================
# CONFIGURE: edit these before running
# ==============================================================================

# Vision-capable models (confirmed on OpenRouter — support image input).
# Used for visual, structured, and judge agents.
VISION_MODELS = [
    "openrouter/anthropic/claude-sonnet-4.6",
    "openrouter/google/gemini-3-flash-preview",
    "openrouter/openai/gpt-5.4",
]

# Text-only models (confirmed no image support on OpenRouter).
# Restricted to the judge agent only.
TEXT_ONLY_MODELS = [
    "openrouter/deepseek/deepseek-v3.2",            # text-only confirmed
    "openrouter/xiaomi/mimo-v2-pro",                # text-only confirmed
    "openrouter/minimax/minimax-m2.5",              # text-only confirmed
]

LIMIT      = 20      # samples per run (increase for reliable numbers)
SPLIT      = "test"  # "test" or "val"

# Fixed evidence model when isolating the judge. Use your best vision model.
JUDGE_BASE = VISION_MODELS[0]

OUTPUT_DIR = Path(__file__).parent / "output"

# ==============================================================================

from benchmark_eval import evaluate, compute_metrics
from chartqa_eval import evaluate_chartqa


def slugify(model: str) -> str:
    name = model.split("/")[-1]
    return re.sub(r"[^\w\-]", "_", name)


def save_result(agent: str, model: str, result: dict) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = slugify(model)
    path = OUTPUT_DIR / f"{agent}__{slug}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return path


def print_summary(all_results: list[dict]) -> None:
    all_agents = ["visual", "structured", "judge"]
    by_agent: dict[str, list] = {}
    for r in all_results:
        by_agent.setdefault(r["agent"], []).append(r)

    print(f"\n{'='*70}")
    print("  BENCHMARK SUMMARY")
    print(f"{'='*70}")

    for agent in all_agents:
        group = by_agent.get(agent, [])
        if not group:
            continue
        print(f"\n  AGENT: {agent.upper()}")

        # Classification
        print(f"\n  Classification:")
        print(f"  {'Model':<42} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}  {'n':>5}")
        print(f"  {'-'*68}")
        for r in sorted(group, key=lambda x: x["metrics"]["f1"], reverse=True):
            m    = r["metrics"]
            slug = slugify(r["model"])[:42]
            star = " ★" if r == sorted(group, key=lambda x: x["metrics"]["f1"], reverse=True)[0] else ""
            print(f"  {slug:<42} {m['accuracy']:>6.3f} {m['precision']:>6.3f} "
                  f"{m['recall']:>6.3f} {m['f1']:>6.3f}  {m['total']:>5}{star}")

        # NLG (visual / structured only)
        nlg_group = [r for r in group if r.get("nlg_metrics")]
        if nlg_group:
            print(f"\n  NLG (ChartCheck-style):")
            print(f"  {'Model':<42} {'BLEU':>5} {'ROUGE':>6} {'MET':>5} {'BERT':>6}  {'n':>5}")
            print(f"  {'-'*68}")
            for r in sorted(nlg_group, key=lambda x: x["nlg_metrics"].get("bertscore", 0), reverse=True):
                m    = r["nlg_metrics"]
                slug = slugify(r["model"])[:42]
                print(f"  {slug:<42} {m['bleu']:>5.1f} {m['rouge_l']:>6.1f} "
                      f"{m['meteor']:>5.1f} {m['bertscore']:>6.1f}  {m['n_nlg']:>5}")

    print(f"\n{'='*70}\n")


def save_excel(all_results: list[dict], chartqa_results: list[dict], path: Path) -> None:
    wb = openpyxl.Workbook()

    header_font  = Font(bold=True, color="FFFFFF")
    header_fill  = PatternFill("solid", fgColor="2E4057")
    center       = Alignment(horizontal="center")

    def style_header_row(ws, row_idx: int) -> None:
        for cell in ws[row_idx]:
            cell.font      = header_font
            cell.fill      = header_fill
            cell.alignment = center

    def autofit(ws) -> None:
        for col in ws.columns:
            max_len = max((len(str(c.value or "")) for c in col), default=10)
            ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + 4, 60)

    # ── Sheet 1: Classification summary ───────────────────────────────────────
    ws1 = wb.active
    ws1.title = "Classification"
    headers = ["Agent", "Model", "Accuracy", "Precision", "Recall", "F1",
               "TP", "FP", "TN", "FN", "Unknown", "Total"]
    ws1.append(headers)
    style_header_row(ws1, 1)
    for r in all_results:
        m = r["metrics"]
        ws1.append([
            r["agent"], r["model"],
            round(m["accuracy"],  3), round(m["precision"], 3),
            round(m["recall"],    3), round(m["f1"],        3),
            m["tp"], m["fp"], m["tn"], m["fn"], m["unknown"], m["total"],
        ])
    autofit(ws1)

    # ── Sheet 2: NLG summary (visual / structured only) ───────────────────────
    ws2 = wb.create_sheet("NLG_ChartFC")
    nlg_headers = ["Agent", "Model", "BLEU", "ROUGE-L", "METEOR", "BERTScore", "n"]
    ws2.append(nlg_headers)
    style_header_row(ws2, 1)
    for r in all_results:
        m = r.get("nlg_metrics")
        if not m:
            continue
        ws2.append([
            r["agent"], r["model"],
            round(m["bleu"],      1), round(m["rouge_l"], 1),
            round(m["meteor"],    1), round(m["bertscore"], 1),
            m["n_nlg"],
        ])
    autofit(ws2)

    # ── Sheet 3: ChartQA (table extraction quality) ───────────────────────────
    ws_cqa = wb.create_sheet("ChartQA")
    cqa_headers = ["Model", "Human Acc%", "Machine Acc%", "Overall Acc%",
                   "Human RNSS", "Machine RNSS", "Overall RNSS", "Total n"]
    ws_cqa.append(cqa_headers)
    style_header_row(ws_cqa, 1)
    # Paper baseline (RelaxedAccuracy only — no RNSS reported in paper)
    ws_cqa.append(["DePlot + FlanT5 few-shot (paper, zero-shot comparable)",
                   28.86, 57.34, 43.1, "-", "-", "-", "-"])
    for r in chartqa_results:
        m = r["metrics"]
        ws_cqa.append([
            r["model"],
            round(m.get("human",   {}).get("relax_acc", 0) * 100, 1),
            round(m.get("machine", {}).get("relax_acc", 0) * 100, 1),
            round(m.get("all",     {}).get("relax_acc", 0) * 100, 1),
            round(m.get("human",   {}).get("rnss", 0), 3),
            round(m.get("machine", {}).get("rnss", 0), 3),
            round(m.get("all",     {}).get("rnss", 0), 3),
            m.get("all", {}).get("n", 0),
        ])
    autofit(ws_cqa)

    # ── Sheet 4: ChartQA raw rows ─────────────────────────────────────────────
    ws_cqa_raw = wb.create_sheet("ChartQA_Raw")
    ws_cqa_raw.append(["Model", "Type", "Query", "Gold Labels",
                        "Predicted Answer", "Correct", "Extracted Table"])
    style_header_row(ws_cqa_raw, 1)
    for r in chartqa_results:
        for row in r.get("rows", []):
            ws_cqa_raw.append([
                r["model"],
                row.get("type", ""),
                row.get("query", ""),
                ", ".join(row.get("gold_labels", [])),
                row.get("predicted_answer", ""),
                row.get("correct", ""),
                row.get("extracted_table", ""),
            ])
    autofit(ws_cqa_raw)

    # ── Sheet 3: Raw per-sample rows ──────────────────────────────────────────
    ws3 = wb.create_sheet("Raw_Rows")
    row_headers = ["Agent", "Model", "Claim", "Ground Truth", "Predicted",
                   "Candidate Answer", "Confidence", "Hypothesis", "Reference"]
    ws3.append(row_headers)
    style_header_row(ws3, 1)
    for r in all_results:
        for row in r.get("rows", []):
            ws3.append([
                r["agent"], r["model"],
                row.get("claim", ""),
                row.get("ground_truth", ""),
                row.get("predicted", ""),
                row.get("candidate_answer", ""),
                row.get("confidence", ""),
                row.get("hypothesis", ""),
                row.get("reference", ""),
            ])
    autofit(ws3)

    wb.save(path)


async def main() -> None:
    vision_agents  = ["visual", "structured", "judge"]
    text_only_agents = ["judge"]  # text-only models can only do judge

    # Build the full run list: (agent, model, base_model)
    runs: list[tuple[str, str, str]] = []
    for agent in vision_agents:
        for model in VISION_MODELS:
            base = JUDGE_BASE if agent == "judge" else model
            runs.append((agent, model, base))
    for model in TEXT_ONLY_MODELS:
        runs.append(("judge", model, JUDGE_BASE))

    print(f"\n{'='*70}")
    print(f"  Chart Verifier - Model Benchmark")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Vision models : {len(VISION_MODELS)}")
    print(f"  Text-only     : {len(TEXT_ONLY_MODELS)}  (judge agent only)")
    print(f"  Total runs    : {len(runs)}")
    print(f"  Samples/run   : {LIMIT}  |  Split: {SPLIT}")
    print(f"  Output        : {OUTPUT_DIR}/")
    print(f"{'='*70}")

    all_results: list[dict] = []

    for run_num, (agent, model, base) in enumerate(runs, 1):
        slug = slugify(model)
        print(f"\n[{run_num}/{len(runs)}]  agent={agent}  model={slug}")

        try:
            result = await evaluate(
                agent      = agent,
                model      = model,
                base_model = base,
                split      = SPLIT,
                limit      = LIMIT,
            )
        except Exception as exc:
            print(f"  FAILED: {exc}")
            print(f"  Skipping {slug} — check that the model slug is correct on OpenRouter.")
            continue

        path = save_result(agent, model, result)
        print(f"  Saved → {path.name}")
        all_results.append(result)

    if not all_results:
        print("\nNo results collected. Check model slugs and API keys.")
        return

    # ── ChartQA: run vision models only (structured agent QA task) ───────────
    chartqa_results: list[dict] = []
    total_cqa = len(VISION_MODELS)
    print(f"\n{'='*70}")
    print(f"  ChartQA Evaluation  ({total_cqa} vision models x {LIMIT} samples)")
    print(f"{'='*70}")
    for cqa_num, model in enumerate(VISION_MODELS, 1):
        slug = slugify(model)
        print(f"\n[{cqa_num}/{total_cqa}]  model={slug}")
        try:
            cqa_result = await evaluate_chartqa(model=model, split=SPLIT, limit=LIMIT)
            out = OUTPUT_DIR / f"chartqa__{slug}.json"
            with open(out, "w") as f:
                json.dump(cqa_result, f, indent=2)
            print(f"  Saved → {out.name}")
            chartqa_results.append(cqa_result)
        except Exception as exc:
            print(f"  FAILED: {exc}")

    summary = {
        "run_at":     datetime.now().isoformat(),
        "split":      SPLIT,
        "limit":      LIMIT,
        "judge_base": JUDGE_BASE,
        "results": [
            {
                "agent":       r["agent"],
                "model":       r["model"],
                "metrics":     r["metrics"],
                "nlg_metrics": r.get("nlg_metrics", {}),
            }
            for r in all_results
        ],
    }
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    excel_path = OUTPUT_DIR / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
    save_excel(all_results, chartqa_results, excel_path)

    print_summary(all_results)
    print(f"  Summary saved → {summary_path}")
    print(f"  Excel saved   → {excel_path}\n")


if __name__ == "__main__":
    asyncio.run(main())
