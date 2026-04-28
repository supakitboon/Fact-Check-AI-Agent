"""
Evaluation: Agent 1 (Claim Analyzer) model ablation

For each model in MODELS, runs Agent 1's relevance classification task:
  - Narrative = chart claim (related) + one fixed unrelated sentence
  - Ground truth: claim sentence → related=true, unrelated sentence → related=false
  - split_sentences is deterministic (regex), so it is called locally before the LLM call

Metrics per model:
  acc_related   — accuracy on claim sentences (should be related=true)
  acc_unrelated — accuracy on filler sentences (should be related=false)
  accuracy      — overall (both sentence types)
  avg_latency_s, avg_tokens, parse_errors

Usage:
  python eval_agent1.py --n 20 --output eval_agent1_results.csv
"""

import argparse
import base64
import csv
import json
import os
import time
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from litellm import completion

load_dotenv(Path(__file__).parent.parent / "chart_verifier" / ".env")

# ── Config ─────────────────────────────────────────────────────────────────────
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")
IMAGE_CACHE    = Path(__file__).parent.parent / "eval_image_cache"
IMAGE_CACHE.mkdir(exist_ok=True)
RATE_DELAY = 2.0  # seconds between API calls

# Models to compare — all must support vision via OpenRouter
MODELS = [
    "openrouter/anthropic/claude-sonnet-4-6",
    "openrouter/anthropic/claude-3.7-sonnet",
    "openrouter/google/gemini-2.5-pro",
    "openrouter/openai/gpt-4o",
]

# Fixed unrelated sentence appended to every narrative to test negative detection
UNRELATED_SENTENCE = "The weather outside is sunny and warm today."

AGENT1_SYSTEM = """You are a Claim Analyzer. You receive a student's chart narrative (already split into sentences) and a chart image.

For each sentence, look at the chart image and decide:
  - related=true  : the sentence is about data, structure, title, axes, or content
                    shown in this specific chart — even if the claim is wrong.
                    When in doubt, choose related=true.
  - related=false : the sentence has no connection to this chart whatsoever.
Set short_circuit=true when related=false, false when related=true.

Output ONLY a JSON array (one object per sentence):
[
  {
    "claim": "<original sentence text, unchanged>",
    "related": <true | false>,
    "short_circuit": <true if unrelated, false if related>,
    "reasoning": "<one sentence explaining the relevance decision>"
  }
]

Do not add preamble or explanation. Return only the JSON array."""


# ── Helpers ────────────────────────────────────────────────────────────────────

def detect_mime(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
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
            print(f"    [warn] download failed: {e}")
            return None
    mime = detect_mime(local.read_bytes()[:12])
    return str(local), mime


def encode_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def split_sentences_local(text: str) -> list[str]:
    """Deterministic sentence splitter — mirrors chart_verifier/tools/text_tools.py."""
    import re
    text = text.strip()
    raw = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in raw if s.strip()]


def extract_json_array(text: str) -> list | None:
    start = text.find("[")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except Exception:
                    return None
    return None


def img_part(img_b64: str, mime: str) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}}


# ── Agent 1 runner ─────────────────────────────────────────────────────────────

def run_agent1(sentences: list[str], img_b64: str, mime: str, model: str) -> tuple[list | None, dict]:
    """
    Calls the model with Agent 1's prompt for the given pre-split sentences.
    Returns (parsed_array_or_None, usage_dict).
    """
    sentence_list = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
    user_text = f"Sentences to classify:\n{sentence_list}"

    time.sleep(RATE_DELAY)
    t0 = time.perf_counter()

    resp = completion(
        model=model,
        messages=[
            {"role": "system", "content": AGENT1_SYSTEM},
            {"role": "user",   "content": [
                {"type": "text", "text": user_text},
                img_part(img_b64, mime),
            ]},
        ],
        max_tokens=1024,
        api_key=OPENROUTER_KEY,
        api_base="https://openrouter.ai/api/v1",
    )

    latency = round(time.perf_counter() - t0, 2)
    content = resp.choices[0].message.content or ""
    usage = {
        "prompt_tokens":     getattr(resp.usage, "prompt_tokens", 0),
        "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
        "latency_s":         latency,
    }
    parsed = extract_json_array(content)
    return parsed, usage


# ── Data loading ───────────────────────────────────────────────────────────────

def load_csv(path: str, n: int) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("chart_img") and row.get("claim"):
                rows.append(row)
    if len(rows) > n:
        step = len(rows) // n
        rows = [rows[i * step] for i in range(n)]
    return rows


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_model_metrics(records: list[dict]) -> dict:
    """Aggregate per-sentence records for one model."""
    related_total = related_correct = 0
    unrelated_total = unrelated_correct = 0
    parse_errors = 0
    latencies = []
    tokens = []

    for r in records:
        if r["parse_error"]:
            parse_errors += 1
            continue

        latencies.append(r["latency_s"])
        tokens.append(r["tokens"])

        if r["gt_related"]:
            related_total += 1
            if r["pred_related"] is True:
                related_correct += 1
        else:
            unrelated_total += 1
            if r["pred_related"] is False:
                unrelated_correct += 1

    total = related_total + unrelated_total
    correct = related_correct + unrelated_correct

    return {
        "accuracy":        correct / total if total else 0.0,
        "acc_related":     related_correct / related_total if related_total else 0.0,
        "acc_unrelated":   unrelated_correct / unrelated_total if unrelated_total else 0.0,
        "parse_errors":    parse_errors,
        "avg_latency_s":   round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "avg_tokens":      round(sum(tokens) / len(tokens), 0) if tokens else 0.0,
        "n_cases":         len({r["idx"] for r in records}),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main(n: int, output: str, models: list[str]) -> None:
    csv_path = Path(__file__).parent.parent / "data" / "chartcheck" / "test.csv"
    rows = load_csv(str(csv_path), n)

    print(f"\nCases   : {len(rows)}")
    print(f"Models  : {len(models)}")
    print(f"Output  : {output}\n")

    all_records: list[dict] = []  # one row per (model, test_case, sentence)

    for model in models:
        model_short = model.split("/")[-1]
        print(f"\n{'='*70}")
        print(f"Model: {model_short}")
        print(f"{'='*70}")

        for idx, row in enumerate(rows, 1):
            claim    = row["claim"].strip()
            img_url  = row["chart_img"].strip()

            img_result = download_image(img_url)
            if img_result is None:
                print(f"  [{idx:>3}] SKIP (image download failed): {claim[:50]}")
                continue

            img_path, mime = img_result
            img_b64 = encode_b64(img_path)

            # Build narrative: claim sentence(s) + one fixed unrelated sentence
            claim_sentences = split_sentences_local(claim)
            sentences = claim_sentences + [UNRELATED_SENTENCE]

            # Ground truth: claim sentences → related, unrelated sentence → not related
            gt_related = [True] * len(claim_sentences) + [False]

            try:
                parsed, usage = run_agent1(sentences, img_b64, mime, model)
            except Exception as e:
                print(f"  [{idx:>3}] ERROR: {e}")
                for sent, gt in zip(sentences, gt_related):
                    all_records.append({
                        "model": model, "idx": idx, "chart_img": img_url,
                        "sentence": sent, "gt_related": gt,
                        "pred_related": None, "reasoning": "",
                        "parse_error": True, "latency_s": 0.0, "tokens": 0,
                    })
                continue

            parse_error = parsed is None
            tokens = usage["prompt_tokens"] + usage["completion_tokens"]

            # Match parsed results back to sentences by position or by text
            pred_map: dict[int, bool | None] = {}
            if parsed:
                for i, item in enumerate(parsed):
                    if i < len(sentences):
                        val = item.get("related")
                        pred_map[i] = bool(val) if isinstance(val, bool) else None

            result_line = []
            for i, (sent, gt) in enumerate(zip(sentences, gt_related)):
                pred = pred_map.get(i)
                match = pred == gt if pred is not None else None
                reasoning = parsed[i].get("reasoning", "") if (parsed and i < len(parsed)) else ""
                all_records.append({
                    "model": model, "idx": idx, "chart_img": img_url,
                    "sentence": sent, "gt_related": gt,
                    "pred_related": pred, "reasoning": reasoning,
                    "parse_error": parse_error,
                    "latency_s": usage["latency_s"],
                    "tokens": tokens,
                })
                mark = "Y" if match else ("N" if match is False else "?")
                result_line.append(f"{'R' if gt else 'U'}:{mark}")

            short = claim[:45] + "..." if len(claim) > 45 else claim
            print(f"  [{idx:>3}] {' '.join(result_line)}  {short}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("AGENT 1 MODEL COMPARISON SUMMARY")
    print("=" * 90)
    print(f"\n{'Model':<35} {'Acc':>6} {'Acc-R':>7} {'Acc-U':>7} {'Err':>5}  "
          f"{'Latency':>8} {'Tokens':>7}")
    print("-" * 80)

    for model in models:
        model_short = model.split("/")[-1]
        records = [r for r in all_records if r["model"] == model]
        m = compute_model_metrics(records)
        print(
            f"{model_short:<35} {m['accuracy']:>6.3f} {m['acc_related']:>7.3f} "
            f"{m['acc_unrelated']:>7.3f} {m['parse_errors']:>5}  "
            f"{m['avg_latency_s']:>7.1f}s {m['avg_tokens']:>7.0f}"
        )

    print("=" * 90)
    print("  Acc   = overall accuracy across both sentence types")
    print("  Acc-R = accuracy on chart-related sentences (should be related=true)")
    print("  Acc-U = accuracy on unrelated sentences (should be related=false)")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    if all_records:
        fieldnames = ["model", "idx", "chart_img", "sentence", "gt_related",
                      "pred_related", "reasoning", "parse_error", "latency_s", "tokens"]
        with open(output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_records)
        print(f"\nDetailed results saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent 1 model ablation study")
    parser.add_argument("--n",      type=int, default=20,
                        help="Test cases per model (default: 20)")
    parser.add_argument("--output", type=str, default="eval_agent1_results.csv",
                        help="Output CSV path")
    parser.add_argument("--models", type=str, default="",
                        help="Comma-separated OpenRouter model IDs (overrides built-in list)")
    args = parser.parse_args()

    chosen_models = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        if args.models else MODELS
    )
    main(args.n, args.output, chosen_models)
