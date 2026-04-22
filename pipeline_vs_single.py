"""
Pipeline vs Single-Model comparison on ChartFC test set.

Runs the same 20 samples through two systems:
  1. Single model  — one vision model, one prompt, direct TRUE/FALSE verdict
  2. Full pipeline — visual agent + structured agent + judge (can use different
                     models per agent once benchmark results are available)

This isolates whether the multi-agent architecture adds value beyond just
picking a good model.

== CONFIGURE HERE =============================================================

  SINGLE_MODEL   — the vision model used for the single-model baseline
  VISUAL_MODEL   — best model for visual agent   (from benchmark results)
  STRUCTURED_MODEL — best model for structured   (from benchmark results)
  AGENT5_MODEL   — tiebreaker model (vision, sees chart directly)

  Set all four to the same model initially. After running run_benchmark.py,
  update each to the winner for that agent.

== RUN ========================================================================

  python pipeline_vs_single.py
  python pipeline_vs_single.py --limit 20 --split test

===============================================================================
"""

import asyncio
import json
import re
import time
from datetime import datetime
from pathlib import Path

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / "chart_verifier" / ".env")

from benchmark_eval import (
    load_samples, download_image, encode_image, remove_temp,
    call_llm, run_visual, run_structured,
    candidate_to_verdict,
    compute_metrics, print_metrics,
    _image_parts, _typed_claims_json,
)
from chart_verifier.agents.claim_typing import INSTRUCTION as TYPING_INSTRUCTION

# ==============================================================================
# CONFIGURE: update with benchmark winners once you have results
# ==============================================================================

# Best model per agent — from benchmark_20260420_2229 (n=20, test split)
SINGLE_MODEL     = "openrouter/anthropic/claude-sonnet-4.6"    # baseline (vision+text)
TYPING_MODEL     = "openrouter/anthropic/claude-sonnet-4.6"    # Agent 2: relevance check
VISUAL_MODEL     = "openrouter/anthropic/claude-sonnet-4.6"    # Agent 3: best visual (F1=0.857)
STRUCTURED_MODEL = "openrouter/openai/gpt-5.4"                 # Agent 4: best structured (F1=0.889)
AGENT5_MODEL     = "openrouter/anthropic/claude-sonnet-4.6"    # Agent 5: tiebreaker
# Note: Agent 1 (Claim Decomposer) is skipped — ChartFC claims are already
# single atomic sentences so decomposition adds no value here.

LIMIT  = 30
SPLIT  = "test"

OUTPUT_DIR = Path(__file__).parent / "output"

# ==============================================================================

SINGLE_MODEL_INSTRUCTION = """
You are a chart fact-checking expert.

You will receive a chart image and a claim about the data in that chart.
Carefully examine the chart and determine whether the claim is TRUE or FALSE.

Return ONLY valid JSON (no preamble):
{
  "verdict": "TRUE" or "FALSE",
  "reasoning": "<one concise sentence explaining your decision>"
}
"""


# ── Single model runner ───────────────────────────────────────────────────────

async def run_single_model(claim: str, img_b64: str, mime: str, model: str) -> tuple[str, str]:
    """Returns (verdict, reasoning)."""
    user_parts = [
        {"type": "text",      "text": f"Claim: {claim}"},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
    ]
    raw = await call_llm(model, SINGLE_MODEL_INSTRUCTION, user_parts)
    try:
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
        obj = json.loads(clean)
        verdict   = str(obj.get("verdict", "")).strip().upper()
        reasoning = str(obj.get("reasoning", "")).strip()
        if verdict not in ("TRUE", "FALSE"):
            verdict = "UNKNOWN"
        return verdict, reasoning
    except Exception:
        for line in raw.splitlines():
            if "TRUE" in line.upper():  return "TRUE",  line.strip()
            if "FALSE" in line.upper(): return "FALSE", line.strip()
        return "UNKNOWN", raw[:200]


# ── Claim typing ──────────────────────────────────────────────────────────────

async def run_typing(claim: str, img_b64: str, mime: str) -> dict:
    """Run Agent 2 (relevance check) with the chart image. Returns {related, reasoning}."""
    claims_json = json.dumps([claim])
    user_parts = [
        {"type": "text", "text": f"Claims to check:\n{claims_json}"},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
    ]
    raw = await call_llm(TYPING_MODEL, TYPING_INSTRUCTION, user_parts)
    try:
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
        arr = json.loads(clean)
        if arr and isinstance(arr[0], dict):
            return arr[0]
    except Exception:
        pass
    return {"claim": claim, "related": True, "reasoning": "fallback"}


# Confidence threshold for ensemble consensus (skip judge when both agree above this)
CONSENSUS_THRESHOLD = 0.85
# Confidence threshold below which we retry the visual agent
RETRY_THRESHOLD     = 0.65

AGENT5_INSTRUCTION = """
You are a Verification Question Agent acting as a tiebreaker.

Agents 3 and 4 have produced conflicting verdicts about a claim. Your job is to
resolve the conflict by going directly back to the chart image.

Step 1 — Generate ONE specific, narrow question whose answer would directly
         settle the claim. Examples:
           "What is the exact percentage shown for X?"
           "Which bar is taller — A or B?"
           "Does the line increase or decrease between 2010 and 2011?"

Step 2 — Answer your own question by carefully reading the chart image.

Step 3 — Based on your answer, give a final verdict on the original claim.
  - If your answer CONFIRMS what the claim says → verdict: "TRUE"
  - If your answer CONTRADICTS what the claim says → verdict: "FALSE"
  - If the chart cannot answer the question → verdict: "FALSE"

Output ONLY valid JSON (no preamble):
{
  "verification_question": "<your targeted question>",
  "answer": "<your answer from the chart>",
  "verdict": "TRUE" or "FALSE",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<1-2 sentences explaining how your answer leads to this verdict>"
}
"""

VISUAL_RETRY_INSTRUCTION = """
You are a Visual Evidence Agent performing a second, more careful analysis.
A previous pass returned low confidence. Re-examine the chart image closely,
focusing specifically on the claim. Pay attention to exact axis values, labels,
legends, and any ambiguous visual elements.

Output ONLY a JSON array (one object):
[
  {
    "claim": "<claim text>",
    "skipped": false,
    "evidence_summary": "<what the chart shows — be precise>",
    "candidate_answer": "<supported | contradicted | partially_supported | insufficient_evidence | unrelated>",
    "confidence": <float 0.0-1.0>,
    "reasoning": "<1-2 sentences>"
  }
]
"""


# ── Agent 5 runner ───────────────────────────────────────────────────────────

async def run_agent5(claim: str, img_b64: str, mime: str,
                     vis: dict, strc: dict) -> dict | None:
    """Tiebreaker — called only when Agent 3 and 4 disagree. Returns verdict dict."""
    context = (
        f"Claim: {claim}\n\n"
        f"Agent 3 (Visual) said: {vis.get('candidate_answer')} "
        f"(confidence {vis.get('confidence')}) — {vis.get('evidence_summary', '')[:150]}\n\n"
        f"Agent 4 (Structured) said: {strc.get('candidate_answer')} "
        f"(confidence {strc.get('confidence')}) — {strc.get('reasoning', '')[:150]}"
    )
    user_parts = [
        {"type": "text",      "text": context},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
    ]
    raw = await call_llm(AGENT5_MODEL, AGENT5_INSTRUCTION, user_parts)
    try:
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
        return json.loads(clean)
    except Exception:
        return None


# ── Pipeline runner ───────────────────────────────────────────────────────────

async def run_pipeline(claim: str, img_b64: str, mime: str) -> tuple[str, dict]:
    """Returns (verdict, details). Runs: typing → visual+structured (parallel) → ensemble/judge."""
    # Agent 2: relevance check (sees claim + image)
    typed = await run_typing(claim, img_b64, mime)

    # Short-circuit: claim is unrelated to this chart
    if not typed.get("related", True):
        return "FALSE", {
            "visual_verdict": "N/A", "structured_verdict": "N/A",
            "judge_verdict": "FALSE",
            "claim_type": "unrelated",
            "short_circuited": True,
            "visual_confidence": None,
            "visual_evidence": "", "structured_evidence": "", "judge_raw": "",
            "pipeline_code_executed": False,
        }

    # Agents 3 & 4 run in parallel — return_exceptions so one failure doesn't kill both
    results = await asyncio.gather(
        run_visual(claim, img_b64, mime, VISUAL_MODEL),
        run_structured(claim, img_b64, mime, STRUCTURED_MODEL),
        return_exceptions=True,
    )
    vis  = results[0] if not isinstance(results[0], Exception) else None
    strc = results[1] if not isinstance(results[1], Exception) else None
    if isinstance(results[0], Exception): print(f"  Visual error  : {results[0]}")
    if isinstance(results[1], Exception): print(f"  Struct error  : {results[1]}")

    # Fix 4: Retry visual if both agents have low confidence
    vis_conf  = (vis.get("confidence")  or 0) if vis  else 0
    strc_conf = (strc.get("confidence") or 0) if strc else 0
    if vis_conf < RETRY_THRESHOLD and strc_conf < RETRY_THRESHOLD:
        initial = vis.get("evidence_summary", "") if vis else ""
        context = (
            f"Claim: {claim}\n"
            f"Initial uncertain evidence: {initial}"
        )
        from benchmark_eval import _image_parts, _parse_first_claim
        retry_parts = _image_parts(img_b64, mime, context)
        retry_raw   = await call_llm(VISUAL_MODEL, VISUAL_RETRY_INSTRUCTION, retry_parts)
        retry_result = _parse_first_claim(retry_raw)
        if retry_result and (retry_result.get("confidence") or 0) > vis_conf:
            vis = retry_result
            vis_conf = retry_result.get("confidence") or 0

    vis_pred  = candidate_to_verdict(vis.get("candidate_answer"))  if vis  else "UNKNOWN"
    strc_pred = candidate_to_verdict(strc.get("candidate_answer")) if strc else "UNKNOWN"

    judge_verdict  = "UNKNOWN"
    judge_raw      = ""
    agent5_used    = False
    agent5_verdict = ""

    # If one agent failed, use the other as both inputs to Agent 5
    if vis and not strc:
        strc = vis
        strc_pred, strc_conf = vis_pred, vis_conf
    elif strc and not vis:
        vis = strc
        vis_pred, vis_conf = strc_pred, strc_conf

    if vis and strc:
        agents_agree = (vis_pred == strc_pred and vis_pred != "UNKNOWN")

        if agents_agree and vis_conf >= CONSENSUS_THRESHOLD and strc_conf >= CONSENSUS_THRESHOLD:
            # Both agree with high confidence — skip judge entirely
            judge_verdict = vis_pred
            judge_raw     = f"[Consensus: {vis_pred} — judge skipped (vis={vis_conf:.2f}, strc={strc_conf:.2f})]"

        else:
            # Agents disagree OR agree with low confidence — Agent 5 tiebreaker
            a5 = await run_agent5(claim, img_b64, mime, vis, strc)
            if a5 and a5.get("verdict") in ("TRUE", "FALSE"):
                agent5_used    = True
                agent5_verdict = a5["verdict"]
                judge_verdict  = a5["verdict"]
                judge_raw      = (f"[Agent 5] Q: {a5.get('verification_question','')} "
                                  f"A: {a5.get('answer','')} → {a5.get('verdict','')} "
                                  f"(conf={a5.get('confidence',0):.2f})")
            else:
                # Agent 5 failed — use higher-confidence agent's verdict
                best = vis_pred if vis_conf >= strc_conf else strc_pred
                judge_verdict = best if best != "UNKNOWN" else (vis_pred if vis_pred != "UNKNOWN" else strc_pred)
                judge_raw     = f"[Agent 5 failed — fallback to higher confidence agent]"

    # Fix 1: structured agent uses "reasoning" not "evidence_summary"
    structured_text = strc.get("reasoning") or strc.get("extracted_table") or "" if strc else ""

    return judge_verdict, {
        "visual_verdict":         vis_pred,
        "structured_verdict":     strc_pred,
        "judge_verdict":          judge_verdict,
        "claim_type":             "related",
        "short_circuited":        False,
        "visual_confidence":      vis_conf,
        "visual_evidence":        (vis.get("evidence_summary") or "")[:200] if vis  else "",
        "structured_evidence":    structured_text[:200],
        "agent5_used":            agent5_used,
        "agent5_verdict":         agent5_verdict,
        "judge_raw":              judge_raw[:300],
        "pipeline_code_executed": strc.get("code_executed", False) if strc else False,
    }


# ── Metrics ───────────────────────────────────────────────────────────────────

def disagreement_analysis(rows: list[dict]) -> dict:
    """Cases where pipeline and single model disagree."""
    pipeline_right_single_wrong = []
    single_right_pipeline_wrong = []
    both_wrong                  = []
    both_right                  = []

    for r in rows:
        gt = r["ground_truth"]
        p  = r["pipeline_verdict"]
        s  = r["single_verdict"]
        p_ok = (p == gt)
        s_ok = (s == gt)

        if p_ok and s_ok:     both_right.append(r)
        elif p_ok and not s_ok: pipeline_right_single_wrong.append(r)
        elif s_ok and not p_ok: single_right_pipeline_wrong.append(r)
        else:                  both_wrong.append(r)

    return {
        "both_right":                  len(both_right),
        "pipeline_right_single_wrong": len(pipeline_right_single_wrong),
        "single_right_pipeline_wrong": len(single_right_pipeline_wrong),
        "both_wrong":                  len(both_wrong),
        "pipeline_only_wins":          pipeline_right_single_wrong,
        "single_only_wins":            single_right_pipeline_wrong,
    }


# ── Excel output ──────────────────────────────────────────────────────────────

def save_excel(rows: list[dict], single_m: dict, pipeline_m: dict,
               disagree: dict, path: Path) -> None:
    wb  = openpyxl.Workbook()
    hf  = Font(bold=True, color="FFFFFF")
    hfill = PatternFill("solid", fgColor="2E4057")
    green = PatternFill("solid", fgColor="C6EFCE")
    red   = PatternFill("solid", fgColor="FFC7CE")
    center = Alignment(horizontal="center")

    def style_header(ws, row=1):
        for cell in ws[row]:
            cell.font = hf; cell.fill = hfill; cell.alignment = center

    def autofit(ws):
        for col in ws.columns:
            w = max((len(str(c.value or "")) for c in col), default=10)
            ws.column_dimensions[get_column_letter(col[0].column)].width = min(w + 4, 60)

    # ── Sheet 1: Summary ──────────────────────────────────────────────────────
    ws1 = wb.active
    ws1.title = "Summary"
    ws1.append(["System", "Model(s)", "Accuracy", "Precision", "Recall", "F1",
                 "TP", "FP", "TN", "FN", "Unknown", "API calls/sample"])
    style_header(ws1)

    single_models  = SINGLE_MODEL.split("/")[-1]
    pipeline_models = (f"Visual:{VISUAL_MODEL.split('/')[-1]} | "
                       f"Struct:{STRUCTURED_MODEL.split('/')[-1]} | "
                       f"Agent5:{AGENT5_MODEL.split('/')[-1]}")

    n_rows          = len(rows)
    pipeline_code_n = sum(1 for r in rows if r.get("pipeline_code_executed"))

    for label, m, models, calls, code_str in [
        ("Single Model",         single_m,   single_models,   1, "N/A (baseline)"),
        ("Multi-Agent Pipeline", pipeline_m, pipeline_models, 3,
         f"{pipeline_code_n}/{n_rows} ({pipeline_code_n/n_rows*100:.0f}%)"),
    ]:
        ws1.append([label, models,
                    round(m["accuracy"],  3), round(m["precision"], 3),
                    round(m["recall"],    3), round(m["f1"],        3),
                    m["tp"], m["fp"], m["tn"], m["fn"], m["unknown"], calls, code_str])

    ws1[1][12].value = "Code Verified"
    ws1[1][12].font  = hf
    ws1[1][12].fill  = hfill
    autofit(ws1)

    # ── Sheet 2: Disagreement analysis ────────────────────────────────────────
    ws2 = wb.create_sheet("Disagreement")
    ws2.append(["Category", "Count", "% of total"])
    style_header(ws2)
    total = len(rows)
    for cat, count in [
        ("Both correct",                     disagree["both_right"]),
        ("Pipeline correct, Single wrong",   disagree["pipeline_right_single_wrong"]),
        ("Single correct, Pipeline wrong",   disagree["single_right_pipeline_wrong"]),
        ("Both wrong",                       disagree["both_wrong"]),
    ]:
        ws2.append([cat, count, f"{count/total*100:.1f}%"])

    # Where pipeline uniquely wins
    ws2.append([])
    ws2.append(["Pipeline wins (single was wrong):"])
    ws2.append(["Claim", "Ground Truth", "Pipeline", "Single", "Visual Evidence"])
    style_header(ws2, ws2.max_row)
    for r in disagree["pipeline_only_wins"]:
        ws2.append([r["claim"][:100], r["ground_truth"],
                    r["pipeline_verdict"], r["single_verdict"],
                    r.get("visual_evidence", "")[:100]])

    # Where single model uniquely wins
    ws2.append([])
    ws2.append(["Single model wins (pipeline was wrong):"])
    ws2.append(["Claim", "Ground Truth", "Single", "Pipeline", "Single Reasoning"])
    style_header(ws2, ws2.max_row)
    for r in disagree["single_only_wins"]:
        ws2.append([r["claim"][:100], r["ground_truth"],
                    r["single_verdict"], r["pipeline_verdict"],
                    r.get("single_reasoning", "")[:100]])
    autofit(ws2)

    # ── Sheet 3: Per-sample detail ────────────────────────────────────────────
    ws3 = wb.create_sheet("Per_Sample")
    headers = ["#", "Claim", "Ground Truth",
               "Single Verdict", "Single OK",
               "Pipeline Verdict", "Pipeline OK", "Pipeline Code",
               "Claim Type", "Short Circuited",
               "Visual Verdict", "Structured Verdict",
               "Single Reasoning", "Visual Evidence", "Structured Evidence"]
    ws3.append(headers)
    style_header(ws3)

    for i, r in enumerate(rows, 1):
        gt = r["ground_truth"]
        s_ok = r["single_verdict"] == gt
        p_ok = r["pipeline_verdict"] == gt
        row_data = [
            i, r["claim"][:100], gt,
            r["single_verdict"],   "YES" if s_ok else "NO",
            r["pipeline_verdict"], "YES" if p_ok else "NO",
            "YES" if r.get("pipeline_code_executed") else "NO",
            r.get("claim_type", ""), r.get("short_circuited", False),
            r.get("visual_verdict", ""), r.get("structured_verdict", ""),
            r.get("single_reasoning", "")[:150],
            r.get("visual_evidence",  "")[:150],
            r.get("structured_evidence", "")[:150],
        ]
        ws3.append(row_data)
        row_idx = ws3.max_row
        ws3.cell(row_idx, 5).fill = green if s_ok else red
        ws3.cell(row_idx, 7).fill = green if p_ok else red
    autofit(ws3)

    wb.save(path)


# ── Main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    print(f"\n{'='*65}")
    print(f"  Pipeline vs Single-Model Comparison")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Samples : {LIMIT}  |  Split: {SPLIT}")
    print(f"\n  Single model : {SINGLE_MODEL.split('/')[-1]}")
    print(f"  Pipeline     : visual={VISUAL_MODEL.split('/')[-1]}")
    print(f"               : structured={STRUCTURED_MODEL.split('/')[-1]}")
    print(f"               : agent5={AGENT5_MODEL.split('/')[-1]}")
    print("="*65)

    samples = load_samples(SPLIT, LIMIT)
    rows: list[dict] = []

    for i, sample in enumerate(samples, 1):
        claim        = sample["claim"].strip()
        ground_truth = sample["label"].strip().upper()
        img_url      = sample["chart_img"].strip()

        print(f"\n[{i}/{len(samples)}] {claim[:70]}...")
        print(f"  Ground truth : {ground_truth}")

        img_path, mime = download_image(img_url)
        if not img_path:
            print("  Skipping: image download failed.")
            continue
        try:
            img_b64 = encode_image(img_path)
        finally:
            remove_temp(img_path)

        # Single model
        try:
            t0 = time.time()
            single_verdict, single_reasoning = await run_single_model(
                claim, img_b64, mime, SINGLE_MODEL)
            single_ms = int((time.time() - t0) * 1000)
        except Exception as exc:
            print(f"  Single error: {exc}")
            single_verdict, single_reasoning, single_ms = "UNKNOWN", "", 0

        # Pipeline
        try:
            t0 = time.time()
            pipeline_verdict, pipeline_details = await run_pipeline(
                claim, img_b64, mime)
            pipeline_ms = int((time.time() - t0) * 1000)
        except Exception as exc:
            print(f"  Pipeline error: {exc}")
            pipeline_verdict, pipeline_details, pipeline_ms = "UNKNOWN", {}, 0

        s_mark = "OK" if single_verdict   == ground_truth else "XX"
        p_mark = "OK" if pipeline_verdict == ground_truth else "XX"
        p_code = " [code]" if pipeline_details.get("pipeline_code_executed") else ""
        print(f"  Single   : {single_verdict:<8} {s_mark}  ({single_ms}ms)")
        print(f"  Pipeline : {pipeline_verdict:<8} {p_mark}{p_code}  ({pipeline_ms}ms)")

        rows.append({
            "claim":            claim,
            "ground_truth":     ground_truth,
            "single_verdict":   single_verdict,
            "single_reasoning": single_reasoning,
            "pipeline_verdict": pipeline_verdict,
            **pipeline_details,
        })

    # Metrics
    single_rows   = [{"ground_truth": r["ground_truth"], "predicted": r["single_verdict"]}
                     for r in rows]
    pipeline_rows = [{"ground_truth": r["ground_truth"], "predicted": r["pipeline_verdict"]}
                     for r in rows]

    single_m   = compute_metrics(single_rows)
    pipeline_m = compute_metrics(pipeline_rows)
    disagree   = disagreement_analysis(rows)

    print_metrics("Single Model",        single_m)
    print_metrics("Multi-Agent Pipeline", pipeline_m)

    pipeline_code_n = sum(1 for r in rows if r.get("pipeline_code_executed"))
    n = len(rows)

    print(f"\n{'='*65}")
    print(f"  DISAGREEMENT ANALYSIS  (n={n})")
    print(f"  Both correct                    : {disagree['both_right']}")
    print(f"  Pipeline correct, Single wrong  : {disagree['pipeline_right_single_wrong']}")
    print(f"  Single correct, Pipeline wrong  : {disagree['single_right_pipeline_wrong']}")
    print(f"  Both wrong                      : {disagree['both_wrong']}")
    print(f"\n  CODE EXECUTION RATE (pipeline Agent 4)")
    print(f"  Code verified   : {pipeline_code_n}/{n} ({pipeline_code_n/n*100:.0f}%) claims")
    print("="*65)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts        = datetime.now().strftime("%Y%m%d_%H%M")
    json_path = OUTPUT_DIR / f"pipeline_vs_single_{ts}.json"
    xlsx_path = OUTPUT_DIR / f"pipeline_vs_single_{ts}.xlsx"

    with open(json_path, "w") as f:
        json.dump({"single_metrics": single_m, "pipeline_metrics": pipeline_m,
                   "disagreement": {k: v for k, v in disagree.items()
                                    if not isinstance(v, list)},
                   "rows": rows}, f, indent=2)

    save_excel(rows, single_m, pipeline_m, disagree, xlsx_path)

    print(f"\n  JSON  saved -> {json_path.name}")
    print(f"  Excel saved -> {xlsx_path.name}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=LIMIT)
    parser.add_argument("--split", default=SPLIT, choices=["test", "test2", "val", "train"])
    args = parser.parse_args()
    LIMIT = args.limit
    SPLIT = args.split
    asyncio.run(main())
