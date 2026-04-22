"""
Evaluate the structured evidence agent on ChartQA using two metrics:

  1. RelaxedAccuracy  — QA accuracy (same as DePlot / MatCha papers)
       Numerical answers: correct if within 5% of gold
       Text answers: exact match (case-insensitive)

  2. RNSS (Relative Number Set Similarity)  — table extraction quality
       Compares numbers extracted from the agent's table against the
       ground truth CSV. This is a direct measure of table extraction
       accuracy, independent of the downstream QA answer.
       Formula: for each gold number, find the closest predicted number,
       score = 1 - |pred - gold| / max(|gold|, 1e-9), averaged over all
       gold numbers.

Both metrics are reported separately for human-written and
machine-generated questions (ChartQA convention).

Paper baselines — RelaxedAccuracy on ChartQA test set:
  DePlot + FlanT5 few-shot:  Human 28.86%  Machine 57.34%  Overall 43.1%
  MatCha (fine-tuned):       Human 38.2%   Machine 74.9%   Overall 56.6%

Usage:
  python chartqa_eval.py --model openrouter/anthropic/claude-sonnet-4.6 --limit 50
  python chartqa_eval.py --compare output/chartqa__claude.json output/chartqa__gpt5.json
"""

import argparse
import asyncio
import base64
import csv
import io
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / "chart_verifier" / ".env")

from benchmark_eval import call_llm

# ── Paths ─────────────────────────────────────────────────────────────────────
CHARTQA_ROOT = Path(__file__).parent / "data" / "chartqa_full" / "ChartQA Dataset"

def _tables_dir(split: str) -> Path:
    return CHARTQA_ROOT / split / "tables"

def _png_dir(split: str) -> Path:
    return CHARTQA_ROOT / split / "png"


# ── Prompt ────────────────────────────────────────────────────────────────────

CHARTQA_INSTRUCTION = """
You are a chart data extraction expert.

You will receive a chart image and a question about the data in that chart.

Step 1 — Extract the COMPLETE data table from the chart:
  - Include all row labels, column headers, and every data value
  - Preserve units (%, $, K, M, etc.)
  - Format as a markdown table

Step 2 — Answer the question using only the extracted table.

Return ONLY valid JSON (no preamble, no markdown fences):
{
  "extracted_table": "<markdown table>",
  "answer": "<concise answer: a number, percentage, or short phrase>"
}

Rules:
- answer must be as short as possible ("14", "Yes", "42%", "United States")
- Do not add explanation or commentary
- If the chart lacks enough data, set answer to "insufficient_evidence"
"""


# ── Metrics ───────────────────────────────────────────────────────────────────

def relaxed_accuracy(prediction: str, gold: str, tolerance: float = 0.05) -> bool:
    pred = prediction.strip().lower().replace(",", "")
    gold = gold.strip().lower().replace(",", "")
    try:
        p = float(re.sub(r"[^\d.\-]", "", pred))
        g = float(re.sub(r"[^\d.\-]", "", gold))
        return abs(p - g) / max(abs(g), 1e-9) <= tolerance
    except ValueError:
        return pred == gold


def _extract_numbers(text: str) -> list[float]:
    return [float(n) for n in re.findall(r"-?\d+\.?\d*", text.replace(",", ""))]


def rnss(pred_text: str, gold_numbers: list[float]) -> float:
    """Relative Number Set Similarity between extracted table text and gold CSV numbers."""
    if not gold_numbers:
        return 1.0
    pred_numbers = _extract_numbers(pred_text)
    if not pred_numbers:
        return 0.0
    scores = []
    for g in gold_numbers:
        closest = min(pred_numbers, key=lambda p: abs(p - g))
        score = max(0.0, 1.0 - abs(closest - g) / max(abs(g), 1e-9))
        scores.append(score)
    return sum(scores) / len(scores)


def _load_gold_numbers(imgname: str, split: str) -> list[float]:
    """Load all numeric values from the ground truth CSV for this chart."""
    stem = Path(imgname).stem
    csv_path = _tables_dir(split) / f"{stem}.csv"
    if not csv_path.exists():
        return []
    numbers = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            for cell in row:
                nums = _extract_numbers(cell)
                numbers.extend(nums)
    return numbers


def compute_metrics(rows: list[dict]) -> dict:
    groups: dict[str, list] = {"human": [], "machine": [], "all": rows}
    for r in rows:
        groups[r["type"]].append(r)

    out = {}
    for name, group in groups.items():
        if not group:
            out[name] = {"relax_acc": 0.0, "rnss": 0.0, "n": 0}
            continue
        relax = sum(1 for r in group if r["correct"]) / len(group)
        avg_rnss = sum(r["rnss"] for r in group) / len(group)
        out[name] = {"relax_acc": relax, "rnss": avg_rnss, "n": len(group)}
    return out


def print_metrics(label: str, metrics: dict) -> None:
    print(f"\n{'='*65}")
    print(f"ChartQA Results -- {label}")
    print(f"  {'Split':<10} {'RelaxAcc':>9} {'RNSS':>7} {'n':>5}")
    print(f"  {'-'*35}")
    for split in ("human", "machine", "all"):
        m = metrics.get(split, {})
        if not m.get("n"):
            continue
        print(f"  {split:<10} {m['relax_acc']*100:>8.1f}%  {m['rnss']:>6.3f}  {m['n']:>5}")

    print(f"\n  RelaxedAccuracy baselines (ChartQA test, zero-shot comparable):")
    print(f"  {'Model':<38} {'Human':>7} {'Machine':>8} {'Overall':>8}")
    print(f"  {'-'*62}")
    print(f"  {'DePlot + FlanT5 few-shot':<38} {'28.9%':>7} {'57.3%':>8} {'43.1%':>8}")
    print(f"  {'-'*62}")
    h = metrics.get("human",   {}).get("relax_acc", 0) * 100
    m = metrics.get("machine", {}).get("relax_acc", 0) * 100
    o = metrics.get("all",     {}).get("relax_acc", 0) * 100
    short = label.split("/")[-1][:38]
    print(f"  {short:<38} {h:>6.1f}%  {m:>6.1f}%  {o:>6.1f}%")
    print("="*65)


# ── Image helper ──────────────────────────────────────────────────────────────

def _img_to_base64(img) -> tuple[str, str]:
    buf = io.BytesIO()
    fmt = getattr(img, "format", None) or "PNG"
    img.save(buf, format=fmt)
    mime = {"PNG": "image/png", "JPEG": "image/jpeg",
            "WEBP": "image/webp"}.get(fmt, "image/png")
    return base64.b64encode(buf.getvalue()).decode(), mime


# ── Answer extraction ─────────────────────────────────────────────────────────

def _parse_response(raw: str) -> tuple[str, str]:
    try:
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
        obj = json.loads(clean)
        return str(obj.get("answer", "")).strip(), str(obj.get("extracted_table", "")).strip()
    except Exception:
        m = re.search(r'"answer"\s*:\s*"([^"]+)"', raw)
        return (m.group(1).strip() if m else ""), ""


# ── Evaluation loop ───────────────────────────────────────────────────────────

async def evaluate_chartqa(model: str, split: str, limit: int | None) -> dict:
    from datasets import load_dataset

    print(f"\n{'='*65}")
    print(f"ChartQA Evaluation  (RelaxedAccuracy + RNSS)")
    print(f"  Model : {model}")
    print(f"  Split : {split}  |  Limit: {limit or 'all'}")
    print(f"  Tables: {_tables_dir(split)}")
    print("="*65)

    if not _tables_dir(split).exists():
        raise FileNotFoundError(
            f"Ground truth tables not found at {_tables_dir(split)}. "
            "Run the download script first."
        )

    ds = load_dataset("ahmed-masry/ChartQA", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    rows: list[dict] = []

    for i, sample in enumerate(ds, 1):
        imgname        = sample["imgname"]
        query          = sample["query"]
        gold_label     = sample["label"]
        qtype          = sample["type"]          # "human" or "machine"
        pil_img        = sample["image"]

        gold_numbers   = _load_gold_numbers(imgname, split)

        print(f"\n[{i}/{len(ds)}] [{qtype}] {query[:65]}")
        print(f"  Gold answer : {gold_label}  |  Gold numbers in CSV: {len(gold_numbers)}")

        img_b64, mime = _img_to_base64(pil_img)
        user_parts = [
            {"type": "text",      "text": f"Question: {query}"},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
        ]

        try:
            raw = await call_llm(model, CHARTQA_INSTRUCTION, user_parts)
            answer, table = _parse_response(raw)
        except Exception as exc:
            print(f"  Error: {exc}")
            answer, table = "", ""

        correct      = relaxed_accuracy(answer, gold_label) if answer else False
        rnss_score   = rnss(table, gold_numbers) if table and gold_numbers else 0.0
        mark         = "OK" if correct else "XX"

        print(f"  Predicted   : {answer!r}  {mark}")
        print(f"  RNSS        : {rnss_score:.3f}")

        rows.append({
            "imgname":          imgname,
            "query":            query,
            "gold_label":       gold_label,
            "gold_numbers":     gold_numbers[:20],  # truncate for JSON size
            "predicted_answer": answer,
            "extracted_table":  table[:500] if table else "",
            "type":             qtype,
            "correct":          correct,
            "rnss":             rnss_score,
        })

    metrics = compute_metrics(rows)
    print_metrics(model, metrics)

    return {
        "task":    "chartqa",
        "model":   model,
        "split":   split,
        "limit":   limit,
        "metrics": metrics,
        "rows":    rows,
    }


# ── Compare ───────────────────────────────────────────────────────────────────

def compare(paths: list[str]) -> None:
    records = [json.load(open(p)) for p in paths]
    print(f"\n{'='*70}")
    print(f"  {'Model':<35} {'H-Acc':>6} {'M-Acc':>6} {'Acc':>6} {'H-RNSS':>7} {'M-RNSS':>7} {'RNSS':>6}")
    print(f"  {'-'*68}")
    for r in sorted(records, key=lambda x: x["metrics"]["all"]["relax_acc"], reverse=True):
        m = r["metrics"]
        tag = r["model"].split("/")[-1][:35]
        print(f"  {tag:<35} "
              f"{m['human']['relax_acc']*100:>5.1f}%  "
              f"{m['machine']['relax_acc']*100:>5.1f}%  "
              f"{m['all']['relax_acc']*100:>5.1f}%  "
              f"{m['human']['rnss']:>6.3f}  "
              f"{m['machine']['rnss']:>6.3f}  "
              f"{m['all']['rnss']:>5.3f}")
    print("="*70)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    default_model = os.getenv("OPENROUTER_MODEL", "openrouter/anthropic/claude-sonnet-4.6")
    parser = argparse.ArgumentParser(
        description="ChartQA: RelaxedAccuracy + RNSS table extraction evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model",   default=default_model)
    parser.add_argument("--split",   default="test", choices=["train", "val", "test"])
    parser.add_argument("--limit",   type=int, default=50)
    parser.add_argument("--output",  default=None)
    parser.add_argument("--compare", nargs="+", metavar="FILE")
    args = parser.parse_args()

    if args.compare:
        compare(args.compare)
        return

    result = asyncio.run(evaluate_chartqa(args.model, args.split, args.limit))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
