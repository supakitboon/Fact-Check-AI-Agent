"""
Evaluation: Agent 2 (Visual Evidence) & Agent 3 (Structured Evidence) model ablation

Loops through multiple vision models, runs both agents on each row of
claim_explanation_verification_pre_tasksets_test_two_V2.csv (columns: chart_img, label, claim),
computes metrics, and saves charts + a CSV of detailed results.

Usage:
  python eval_agent23_models.py
  python eval_agent23_models.py --n 50 --output results.csv
"""

import argparse
import base64
import concurrent.futures
import csv
import json
import os
import time
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent.parent / "chart_verifier" / ".env")

# ── Config ─────────────────────────────────────────────────────────────────────
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")
_client = OpenAI(api_key=OPENROUTER_KEY, base_url="https://openrouter.ai/api/v1")
IMAGE_CACHE    = Path(__file__).parent.parent / "eval_image_cache"
IMAGE_CACHE.mkdir(exist_ok=True)
RATE_DELAY  = 0.0
MAX_TOKENS  = 1024

MODELS = [
    "x-ai/grok-4.1-fast"
    #"google/gemini-3-flash-preview",
    #"moonshotai/kimi-k2.6",
    #"qwen/qwen3.6-plus"
]

CSV_PATH = (
    Path(__file__).parent.parent
    / "data"
    / "claim_explanation_verification_pre_tasksets_test_two_V2.csv"
)

# ── Agent prompts ──────────────────────────────────────────────────────────────
AGENT2_SYSTEM = """You are a Chart Reader Agent. You receive a chart image and a factual claim.

Examine the chart image visually and decide:
  correct   — the chart supports the claim
  incorrect — the chart does not support the claim

"unrelated" is NOT a valid verdict. You MUST choose correct or incorrect.

Confidence scoring rules:
  0.9-1.0 — labels, bars, lines, or colors clearly and directly confirm or deny the claim
  0.6-0.8 — claim is mostly verifiable visually but requires some estimation or interpretation
  0.3-0.5 — chart is ambiguous, crowded, or hard to read for this specific claim
  0.0-0.2 — chart does not contain enough visual information to evaluate this claim

Output ONLY a JSON object — no markdown fences, no extra text:
{
  "verdict": "<correct | incorrect>",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<1-2 sentences>"
}

Do not add preamble. Return only the JSON object."""

AGENT3_SYSTEM = """You are a Structured Evidence Agent. You receive a chart image and a factual claim.

Step 1 — Extract a data table:
  Read the chart and extract its data into a structured pseudo-table
  (e.g., rows of "Category | Year | Value"). Estimate from the chart scale if exact
  values are not readable.

Step 2 — Evaluate the claim against the table:
  Decide:
    correct   — the extracted data confirms the claim
    incorrect — the extracted data contradicts the claim

"unrelated" is NOT a valid verdict. You MUST choose correct or incorrect.

Confidence scoring rules:
  0.9-1.0 — exact values read directly from the chart confirm or deny the claim with no ambiguity
  0.6-0.8 — values estimated from scale are close enough to verify the claim with reasonable certainty
  0.3-0.5 — values are difficult to read precisely; significant estimation was needed
  0.0-0.2 — could not extract enough data from the chart to evaluate the claim reliably

Output ONLY a JSON object — no markdown fences, no extra text:
{
  "extracted_table": "<markdown table>",
  "verdict": "<correct | incorrect>",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<1-2 sentences>"
}

Do not add preamble. Return only the JSON object."""


# ── Helpers ────────────────────────────────────────────────────────────────────

def detect_mime(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":         return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":   return "image/png"
    if data[:6] in (b"GIF87a", b"GIF89a"): return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP": return "image/webp"
    return "image/jpeg"


def download_image(url: str) -> tuple[str, str] | None:
    fname = url.split("/")[-1].split("?")[0]
    local = IMAGE_CACHE / fname
    if not local.exists():
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "eval/1.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                local.write_bytes(r.read())
        except Exception as e:
            print(f"  [warn] download failed: {e}")
            return None
    mime = detect_mime(local.read_bytes()[:12])
    return str(local), mime


def encode_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def img_part(img_b64: str, mime: str) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}}


def extract_json_object(text: str) -> dict | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":   depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:    return json.loads(text[start: i + 1])
                except: return None
    return None


def call_model(messages: list, model: str) -> tuple[str, dict]:
    resp = _client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=MAX_TOKENS,
    )
    usage = {
        "prompt_tokens":     resp.usage.prompt_tokens if resp.usage else 0,
        "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
    }
    return resp.choices[0].message.content or "", usage


def map_verdict(v: str) -> str:
    v = v.lower()
    if v == "correct":   return "supported"
    if v == "incorrect": return "contradicted"
    return v


def verdict_to_bool(verdict: str) -> bool | None:
    v = verdict.lower()
    if v in ("supported", "correct"):      return True
    if v in ("contradicted", "incorrect"): return False
    return None


# ── Agent runner ───────────────────────────────────────────────────────────────

def run_agent(system_prompt: str, claim: str, img_b64: str, mime: str, model: str) -> dict:
    t0 = time.perf_counter()
    try:
        raw, usage = call_model(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": f"Claim: {claim}"},
                    img_part(img_b64, mime),
                ]},
            ],
            model=model,
        )
        obj     = extract_json_object(raw) or {}
        verdict = map_verdict(obj.get("verdict", "unknown"))
        if verdict not in ("supported", "contradicted"):
            verdict = "contradicted"
        return {
            "verdict":     verdict,
            "confidence":  float(obj.get("confidence", 0.0)),
            "reasoning":   obj.get("reasoning", raw[:100]),
            "latency_s":   round(time.perf_counter() - t0, 2),
            "tokens":      usage["prompt_tokens"] + usage["completion_tokens"],
            "parse_error": obj == {},
        }
    except Exception as e:
        return {
            "verdict": "error", "confidence": 0.0, "reasoning": str(e),
            "latency_s": round(time.perf_counter() - t0, 2),
            "tokens": 0, "parse_error": True,
        }


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(rows: list[dict]) -> dict:
    tp = fp = tn = fn = parse_err = 0
    latencies, tokens, confidences = [], [], []

    for r in rows:
        gt   = r["ground_truth"]
        pred = verdict_to_bool(r["verdict"])
        if pred is None or r.get("parse_error"):
            parse_err += 1
            continue
        latencies.append(r["latency_s"])
        tokens.append(r["tokens"])
        confidences.append(r["confidence"])
        if gt and pred:       tp += 1
        elif gt and not pred: fn += 1
        elif not gt and pred: fp += 1
        else:                 tn += 1

    total = tp + fp + tn + fn
    acc   = (tp + tn) / total if total else 0
    prec  = tp / (tp + fp)    if (tp + fp) else 0
    rec   = tp / (tp + fn)    if (tp + fn) else 0
    f1    = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

    def acc_subset(subset):
        if not subset: return 0.0
        c = sum(1 for r in subset
                if verdict_to_bool(r["verdict"]) == r["ground_truth"]
                and verdict_to_bool(r["verdict"]) is not None)
        return c / len(subset)

    true_rows  = [r for r in rows if r["ground_truth"] is True]
    false_rows = [r for r in rows if r["ground_truth"] is False]

    return {
        "accuracy":       acc,
        "precision":      prec,
        "recall":         rec,
        "f1":             f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "parse_errors":   parse_err,
        "acc_true":       acc_subset(true_rows),
        "acc_false":      acc_subset(false_rows),
        "avg_latency_s":  round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "avg_tokens":     round(sum(tokens) / len(tokens), 0) if tokens else 0.0,
        "avg_confidence": round(sum(confidences) / len(confidences), 3) if confidences else 0.0,
        "n": total,
    }


# ── Data loading ───────────────────────────────────────────────────────────────

def load_dataset(path: str, n: int | None) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            img   = row.get("chart_img", "").strip()
            label = row.get("label", "").strip()
            claim = row.get("claim", "").strip()
            if img and label and claim:
                rows.append({
                    "chart_img":    img,
                    "ground_truth": label.upper() == "TRUE",
                    "claim":        claim,
                })
    if n and len(rows) > n:
        step = len(rows) // n
        rows = [rows[i * step] for i in range(n)]
    return rows


# ── Charts ─────────────────────────────────────────────────────────────────────

AGENT_COLORS = {"Agent 2 (Visual)": "#4C8BF5", "Agent 3 (Structured)": "#F5A623"}
AGENTS       = ["Agent 2 (Visual)", "Agent 3 (Structured)"]


def _bar_group(ax, models, agents, summary, metric, value_fmt=".2f"):
    x      = np.arange(len(models))
    width  = 0.35
    for i, agent in enumerate(agents):
        subset = summary[summary["agent"] == agent].set_index("model")
        vals   = [subset.loc[m, metric] if m in subset.index else 0 for m in models]
        bars   = ax.bar(x + (i - 0.5) * width, vals, width, label=agent,
                        color=AGENT_COLORS[agent], alpha=0.88)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:{value_fmt}}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([m[:20] for m in models], rotation=18, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)


def plot_metrics(summary: pd.DataFrame, models: list[str], out_dir: Path) -> None:
    # ── 1. Accuracy / Precision / Recall / F1 ─────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Agent 2 vs Agent 3 — Model Comparison", fontsize=15, fontweight="bold")
    for ax, metric in zip(axes.flat, ["accuracy", "precision", "recall", "f1"]):
        _bar_group(ax, models, AGENTS, summary, metric)
        ax.set_title(metric.capitalize(), fontsize=12)
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    plt.tight_layout()
    path = out_dir / "agent23_metrics.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # ── 2. Accuracy by ground-truth class ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Accuracy Breakdown by Ground-Truth Label", fontsize=13, fontweight="bold")
    for ax, metric, title in zip(
        axes,
        ["acc_true", "acc_false"],
        ["Acc on TRUE claims (should predict: supported)",
         "Acc on FALSE claims (should predict: contradicted)"],
    ):
        _bar_group(ax, models, AGENTS, summary, metric)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    plt.tight_layout()
    path = out_dir / "agent23_acc_breakdown.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # ── 3. Latency & token usage ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Cost Proxies: Latency & Token Usage", fontsize=13, fontweight="bold")
    for ax, metric, ylabel in zip(axes,
                                   ["avg_latency_s", "avg_tokens"],
                                   ["Avg Latency (s)", "Avg Tokens"]):
        _bar_group(ax, models, AGENTS, summary, metric, value_fmt=".1f")
        ax.set_title(ylabel, fontsize=11)
        ax.set_ylabel(ylabel)
    plt.tight_layout()
    path = out_dir / "agent23_cost.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # ── 4. Confidence distribution (box plot) ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle("Confidence Score Distribution", fontsize=13, fontweight="bold")
    for ax, agent_key, title in zip(
        axes,
        ["agent2", "agent3"],
        ["Agent 2 (Visual)", "Agent 3 (Structured)"],
    ):
        data_per_model = []
        labels         = []
        for model, data in _results_ref.items():
            confs = [r["confidence"] for r in data[agent_key] if not r.get("parse_error")]
            data_per_model.append(confs)
            labels.append(model.split("/")[-1][:18])
        ax.boxplot(data_per_model, labels=labels, patch_artist=True,
                   boxprops=dict(facecolor=AGENT_COLORS[title], alpha=0.7))
        ax.set_title(title, fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Confidence")
        ax.tick_params(axis="x", rotation=18, labelsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = out_dir / "agent23_confidence_dist.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # ── 5. F1 heatmap ────────────────────────────────────────────────────────
    pivot = summary.pivot(index="agent", columns="model", values="f1")
    pivot.columns = [c[:20] for c in pivot.columns]
    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=18, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title("F1 Score Heatmap (Agent × Model)", fontsize=12, fontweight="bold")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=10, color="black" if 0.3 < val < 0.85 else "white")
    plt.tight_layout()
    path = out_dir / "agent23_f1_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# module-level reference so plot_metrics can access raw results for box plot
_results_ref: dict = {}


# ── Main ───────────────────────────────────────────────────────────────────────

def main(n: int | None, output: str, models: list[str]) -> None:
    assert OPENROUTER_KEY, "Set OPENROUTER_API_KEY in chart_verifier/.env"

    rows = load_dataset(str(CSV_PATH), n)
    print(f"\nCSV     : {CSV_PATH.name}")
    print(f"Cases   : {len(rows)}")
    print(f"Models  : {len(models)}")
    print(f"Output  : {output}\n")

    results: dict[str, dict[str, list[dict]]] = {}

    for model in models:
        model_short = model.split("/")[-1]
        print(f"\n{'='*70}")
        print(f"Model: {model_short}")
        print(f"{'='*70}")

        a2_rows: list[dict] = []
        a3_rows: list[dict] = []

        for idx, row in enumerate(rows, 1):
            claim   = row["claim"]
            img_url = row["chart_img"]
            gt      = row["ground_truth"]
            short   = claim[:45] + "..." if len(claim) > 45 else claim

            img_result = download_image(img_url)
            if img_result is None:
                print(f"  [{idx:>3}] SKIP  {short}")
                continue

            img_path, mime = img_result
            img_b64 = encode_b64(img_path)

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                f2 = ex.submit(run_agent, AGENT2_SYSTEM, claim, img_b64, mime, model)
                f3 = ex.submit(run_agent, AGENT3_SYSTEM, claim, img_b64, mime, model)
                a2, a3 = f2.result(), f3.result()

            a2.update({"idx": idx, "claim": claim, "chart_img": img_url, "ground_truth": gt})
            a2_rows.append(a2)
            a3.update({"idx": idx, "claim": claim, "chart_img": img_url, "ground_truth": gt})
            a3_rows.append(a3)

            a2_mark = "Y" if verdict_to_bool(a2["verdict"]) == gt else "N"
            a3_mark = "Y" if verdict_to_bool(a3["verdict"]) == gt else "N"
            print(f"  [{idx:>3}] A2:{a2['verdict'][:4]}({a2['confidence']:.2f}){a2_mark}  "
                  f"A3:{a3['verdict'][:4]}({a3['confidence']:.2f}){a3_mark}  {short}")

        results[model] = {"agent2": a2_rows, "agent3": a3_rows}

    global _results_ref
    _results_ref = results

    # ── Build summary ──────────────────────────────────────────────────────────
    records = []
    for model, data in results.items():
        model_short = model.split("/")[-1]
        for agent_key, agent_label in (("agent2", "Agent 2 (Visual)"),
                                        ("agent3", "Agent 3 (Structured)")):
            m = compute_metrics(data[agent_key])
            records.append({"model": model_short, "agent": agent_label, **m})

    summary_df = pd.DataFrame(records)

    # ── Print summary table ────────────────────────────────────────────────────
    print("\n" + "="*110)
    print("AGENT 2 & AGENT 3 — MODEL COMPARISON SUMMARY")
    print("="*110)
    header = (f"{'Model':<24} {'Agent':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}  "
              f"{'AccT':>6} {'AccF':>6}  {'Conf':>6} {'Err':>4}  {'Lat(s)':>7} {'Tokens':>7}")
    print(header)
    print("-"*110)
    for _, row in summary_df.iterrows():
        print(
            f"{row['model']:<24} {row['agent']:<22} "
            f"{row['accuracy']:>6.3f} {row['precision']:>6.3f} {row['recall']:>6.3f} {row['f1']:>6.3f}  "
            f"{row['acc_true']:>6.3f} {row['acc_false']:>6.3f}  "
            f"{row['avg_confidence']:>6.3f} {int(row['parse_errors']):>4}  "
            f"{row['avg_latency_s']:>7.1f} {int(row['avg_tokens']):>7}"
        )
    print("="*110)
    print("AccT = accuracy on TRUE claims | AccF = accuracy on FALSE claims | Conf = avg confidence")

    for agent_label in AGENTS:
        sub  = summary_df[summary_df["agent"] == agent_label]
        best = sub.loc[sub["f1"].idxmax()]
        print(f"\nBest {agent_label}: {best['model']}  (F1 = {best['f1']:.3f})")

    # ── Charts ─────────────────────────────────────────────────────────────────
    out_dir = Path(output).parent if Path(output).parent != Path(".") else Path(".")
    models_short = [m.split("/")[-1] for m in models]
    summary_df_short = summary_df.copy()  # model column already uses short names
    plot_metrics(summary_df_short, models_short, out_dir)

    # ── Save detailed CSV ──────────────────────────────────────────────────────
    detail_rows = []
    for model, data in results.items():
        for agent_key in ("agent2", "agent3"):
            for r in data[agent_key]:
                detail_rows.append({"model": model, "agent": agent_key, **r})

    fieldnames = list(detail_rows[0].keys()) if detail_rows else []
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(detail_rows)
    print(f"\nDetailed results saved to {output}  ({len(detail_rows)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent 2 & 3 model ablation study")
    parser.add_argument("--n",      type=int, default=200,
                        help="Test cases to evaluate (default: 100, use 0 for all)")
    parser.add_argument("--output", type=str, default="eval_agent23_results.csv",
                        help="Output CSV path")
    parser.add_argument("--models", type=str, default="",
                        help="Comma-separated OpenRouter model IDs (overrides built-in list)")
    args = parser.parse_args()

    chosen_models = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        if args.models else MODELS
    )
    main(args.n or None, args.output, chosen_models)