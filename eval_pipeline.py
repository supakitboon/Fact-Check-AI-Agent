"""
Evaluation: Ablation study across 3 fact-checking systems

  System 1 — Visual only:     Agent 2 alone (visual chart inspection)
  System 2 — Structured only: Agent 3 alone (data table extraction)
  System 3 — Pipeline:        Agent 2 + Agent 3 → Agent 4 (combined)

Verdict mapping (honest — unrelated counts as wrong):
  supported / correct   → True  (claim verified)
  contradicted / incorrect / unrelated → False (claim not verified)
  error / parse failure → None  (excluded from all metrics, reported separately)

Ground truth:
  label=TRUE  → True  (claim is correct about the chart)
  label=FALSE → False (claim is wrong about the chart)

Usage:
  python eval_pipeline.py --n 20 --output eval_results.csv
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

load_dotenv(Path(__file__).parent / "chart_verifier" / ".env")

# ── Config ─────────────────────────────────────────────────────────────────────
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL       = os.getenv("OPENROUTER_VISION_MODEL", "openrouter/anthropic/claude-sonnet-4-6")
OPUS_MODEL  = os.getenv("OPENROUTER_OPUS_MODEL",   "openrouter/anthropic/claude-opus-4-7")
IMAGE_CACHE = Path(__file__).parent / "eval_image_cache"
IMAGE_CACHE.mkdir(exist_ok=True)
RATE_DELAY = 2.0  # seconds between API calls

MIME_MAP = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png",  ".webp": "image/webp",
    ".gif": "image/gif",
}

SYSTEMS = ["visual_only", "structured_only", "pipeline"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def detect_mime(data: bytes) -> str:
    """Detect image MIME type from magic bytes, ignoring file extension."""
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"  # safe fallback


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


def call_model(messages: list, max_tokens: int = 1024, model: str = MODEL) -> tuple[str, dict]:
    """Returns (content, usage_dict)."""
    resp = completion(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        api_key=OPENROUTER_KEY,
        api_base="https://openrouter.ai/api/v1",
    )
    usage = {
        "prompt_tokens":     getattr(resp.usage, "prompt_tokens", 0),
        "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
    }
    return resp.choices[0].message.content or "", usage


def extract_json_object(text: str) -> dict | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except Exception:
                    return None
    return None


def img_part(img_b64: str, mime: str) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}}


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

AGENT4_SYSTEM = """You are the Verdict Feedback Agent. You receive a factual claim and outputs from two evidence agents.

Resolve the final verdict:
  Case A — both agents agree AND avg confidence >= 0.70 → use shared verdict
  Case B — disagree OR avg confidence < 0.70:
    numerical claims (numbers, %, quantities) → prefer Agent 3 (structured)
    visual claims (trends, patterns, directions) → prefer Agent 2 (visual)

Map: correct → supported,  incorrect → contradicted

Output ONLY a JSON object — no markdown fences, no extra text:
{
  "verdict": "<supported | contradicted | unrelated>",
  "feedback": "<1-2 sentence explanation>"
}"""

AGENT4_TIEBREAKER_SYSTEM = """You are a Senior Fact-Check Arbiter. Two agents tried to verify a claim against a chart but had low confidence or disagreed with each other.

You will receive:
- The original claim
- The chart image
- Each agent's verdict, confidence score, and reasoning

Your task:
1. Read what each agent found uncertain or conflicting
2. Look carefully at the exact part of the chart relevant to the claim
3. Give a definitive binary verdict — you MUST commit to supported or contradicted

Output ONLY a JSON object — no markdown fences, no extra text:
{
  "verdict": "<supported | contradicted>",
  "confidence": <float 0.0-1.0>,
  "focus": "<the specific chart element you examined to resolve the uncertainty>",
  "reasoning": "<1-2 sentences explaining your decision>"
}"""

# ── Systems ────────────────────────────────────────────────────────────────────

def run_visual_only(claim: str, img_b64: str, mime: str) -> dict:
    t0 = time.perf_counter()
    time.sleep(RATE_DELAY)
    raw, usage = call_model([
        {"role": "system", "content": AGENT2_SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": f"Claim: {claim}"},
            img_part(img_b64, mime),
        ]},
    ])
    obj = extract_json_object(raw) or {}
    verdict = _map_agent_verdict(obj.get("verdict", "unknown"))
    if verdict not in ("supported", "contradicted"):
        verdict = "contradicted"
    return {
        "verdict":    verdict,
        "confidence": obj.get("confidence", 0.0),
        "reasoning":  obj.get("reasoning", raw[:120]),
        "latency_s":  round(time.perf_counter() - t0, 2),
        "tokens":     usage["prompt_tokens"] + usage["completion_tokens"],
        "api_calls":  1,
    }


def run_structured_only(claim: str, img_b64: str, mime: str) -> dict:
    t0 = time.perf_counter()
    time.sleep(RATE_DELAY)
    raw, usage = call_model([
        {"role": "system", "content": AGENT3_SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": f"Claim: {claim}"},
            img_part(img_b64, mime),
        ]},
    ])
    obj = extract_json_object(raw) or {}
    verdict = _map_agent_verdict(obj.get("verdict", "unknown"))
    # Enforce binary — A3 must commit to supported or contradicted
    if verdict not in ("supported", "contradicted"):
        verdict = "contradicted"  # safe fallback: refuse to confirm unsupported claims
    return {
        "verdict":    verdict,
        "confidence": obj.get("confidence", 0.0),
        "reasoning":  obj.get("reasoning", raw[:120]),
        "latency_s":  round(time.perf_counter() - t0, 2),
        "tokens":     usage["prompt_tokens"] + usage["completion_tokens"],
        "api_calls":  1,
    }


def run_weighted(claim: str, img_b64: str, mime: str) -> dict:
    """Run A2 + A3; escalate to Agent 4 (Opus) if confidence < TIEBREAK_THRESHOLD."""
    t0 = time.perf_counter()
    total_tokens = 0

    # Agent 2 — visual
    time.sleep(RATE_DELAY)
    a2_raw, a2_usage = call_model([
        {"role": "system", "content": AGENT2_SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": f"Claim: {claim}"},
            img_part(img_b64, mime),
        ]},
    ])
    a2 = extract_json_object(a2_raw) or {"verdict": "unknown", "confidence": 0.0, "reasoning": ""}
    a2_v = _map_agent_verdict(a2.get("verdict", "unknown"))
    if a2_v not in ("supported", "contradicted"):
        a2_v = "contradicted"
    a2_c = float(a2.get("confidence", 0.0))
    total_tokens += a2_usage["prompt_tokens"] + a2_usage["completion_tokens"]

    # Agent 3 — structured
    time.sleep(RATE_DELAY)
    a3_raw, a3_usage = call_model([
        {"role": "system", "content": AGENT3_SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": f"Claim: {claim}"},
            img_part(img_b64, mime),
        ]},
    ])
    a3 = extract_json_object(a3_raw) or {"verdict": "unknown", "confidence": 0.0, "reasoning": ""}
    a3_v = _map_agent_verdict(a3.get("verdict", "unknown"))
    if a3_v not in ("supported", "contradicted"):
        a3_v = "contradicted"
    a3_c = float(a3.get("confidence", 0.0))
    total_tokens += a3_usage["prompt_tokens"] + a3_usage["completion_tokens"]

    avg_conf, needs_tiebreak = _avg_confidence(a2_v, a2_c, a3_v, a3_c)

    if not needs_tiebreak:
        # Agents agree with sufficient confidence — commit directly
        w_verdict = "supported" if avg_conf > 0 else "contradicted"
        w_conf    = round(abs(avg_conf), 4)
        tiebreak  = False
        api_calls = 2
    else:
        # Low confidence or disagreement — escalate to Agent 4 (Opus 4.7)
        time.sleep(RATE_DELAY)
        a4_user = (
            f"Claim: {claim}\n\n"
            f"Agent 2 (Visual):     verdict={a2_v}, confidence={a2_c:.2f}, "
            f"reasoning={a2.get('reasoning', '').strip()}\n\n"
            f"Agent 3 (Structured): verdict={a3_v}, confidence={a3_c:.2f}, "
            f"reasoning={a3.get('reasoning', '').strip()}\n\n"
            f"Avg confidence: {avg_conf:.3f} (below threshold {TIEBREAK_THRESHOLD}) — "
            f"please examine the chart carefully and give a definitive verdict."
        )
        a4_raw, a4_usage = call_model(
            [
                {"role": "system", "content": AGENT4_TIEBREAKER_SYSTEM},
                {"role": "user", "content": [
                    {"type": "text", "text": a4_user},
                    img_part(img_b64, mime),
                ]},
            ],
            model=OPUS_MODEL,
        )
        a4 = extract_json_object(a4_raw) or {"verdict": "contradicted", "confidence": 0.0}
        a4_v = _map_agent_verdict(a4.get("verdict", "contradicted"))
        if a4_v not in ("supported", "contradicted"):
            a4_v = "contradicted"
        total_tokens += a4_usage["prompt_tokens"] + a4_usage["completion_tokens"]

        w_verdict = a4_v
        w_conf    = round(float(a4.get("confidence", 0.0)), 4)
        tiebreak  = True
        api_calls = 3

    return {
        "a2_verdict":    a2_v,
        "a2_confidence": a2_c,
        "a3_verdict":    a3_v,
        "a3_confidence": a3_c,
        "verdict":       w_verdict,
        "confidence":    w_conf,
        "tiebreak":      tiebreak,
        "latency_s":     round(time.perf_counter() - t0, 2),
        "tokens":        total_tokens,
        "api_calls":     api_calls,
    }


def run_pipeline(claim: str, img_b64: str, mime: str) -> dict:
    t0 = time.perf_counter()
    total_tokens = 0

    # Agent 2 — visual evidence
    time.sleep(RATE_DELAY)
    a2_raw, a2_usage = call_model([
        {"role": "system", "content": AGENT2_SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": f"Claim: {claim}"},
            img_part(img_b64, mime),
        ]},
    ])
    a2 = extract_json_object(a2_raw) or {"verdict": "unknown", "confidence": 0.0, "reasoning": ""}
    total_tokens += a2_usage["prompt_tokens"] + a2_usage["completion_tokens"]

    # Agent 3 — structured evidence
    time.sleep(RATE_DELAY)
    a3_raw, a3_usage = call_model([
        {"role": "system", "content": AGENT3_SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": f"Claim: {claim}"},
            img_part(img_b64, mime),
        ]},
    ])
    a3 = extract_json_object(a3_raw) or {"verdict": "unknown", "confidence": 0.0, "reasoning": ""}
    total_tokens += a3_usage["prompt_tokens"] + a3_usage["completion_tokens"]

    # Agent 4 — verdict (text only, no image)
    time.sleep(RATE_DELAY)
    a4_raw, a4_usage = call_model([
        {"role": "system", "content": AGENT4_SYSTEM},
        {"role": "user", "content": (
            f"Claim: {claim}\n\n"
            f"Agent 2 (Visual):\n{json.dumps(a2, indent=2)}\n\n"
            f"Agent 3 (Structured):\n{json.dumps(a3, indent=2)}"
        )},
    ])
    a4 = extract_json_object(a4_raw) or {"verdict": "unknown", "feedback": ""}
    total_tokens += a4_usage["prompt_tokens"] + a4_usage["completion_tokens"]

    a2_v = _map_agent_verdict(a2.get("verdict", "unknown"))
    a3_v = _map_agent_verdict(a3.get("verdict", "unknown"))

    return {
        "verdict":       a4.get("verdict", "unknown"),
        "feedback":      a4.get("feedback", ""),
        "a2_verdict":    a2_v,
        "a2_confidence": a2.get("confidence", 0.0),
        "a3_verdict":    a3_v,
        "a3_confidence": a3.get("confidence", 0.0),
        "a2_a3_agree":   a2_v == a3_v and a2_v not in ("unknown", "error"),
        "latency_s":     round(time.perf_counter() - t0, 2),
        "tokens":        total_tokens,
        "api_calls":     3,
    }


TIEBREAK_THRESHOLD = 0.5  # if net confidence < this, escalate to Agent 4 tiebreaker

def _avg_confidence(v1: str, c1: float, v2: str, c2: float) -> tuple[float, bool]:
    """Returns (avg_conf, needs_tiebreak).

    avg_conf = mean of signed confidence scores (supported → +c, contradicted → -c).
    needs_tiebreak is True when |avg_conf| < TIEBREAK_THRESHOLD,
    meaning agents disagree or are both low-confidence.
    """
    scores = []
    for v, c in [(v1, c1), (v2, c2)]:
        if v == "supported":
            scores.append(+c)
        elif v == "contradicted":
            scores.append(-c)

    if not scores:
        return 0.0, True

    avg_conf = sum(scores) / len(scores)
    return avg_conf, abs(avg_conf) < TIEBREAK_THRESHOLD



def _map_agent_verdict(v: str) -> str:
    """Normalise agent 2/3 correct/incorrect → supported/contradicted."""
    v = v.lower()
    if v == "correct":
        return "supported"
    if v == "incorrect":
        return "contradicted"
    return v  # pass through supported/contradicted/unrelated/unknown


# ── Metrics ────────────────────────────────────────────────────────────────────

def verdict_to_bool(verdict: str) -> bool | None:
    """
    Honest mapping — unrelated counts as NOT supported (False).
    Only parse errors return None (excluded from metrics).
    """
    v = verdict.lower()
    if v in ("supported", "correct"):
        return True
    if v in ("contradicted", "incorrect", "unrelated"):
        return False
    return None  # error / unknown → parse failure


def compute_metrics(rows: list[dict], key: str) -> dict:
    tp = fp = tn = fn = parse_err = 0
    unrelated_count = 0

    for r in rows:
        gt   = r["ground_truth"]
        raw  = r.get(f"{key}_verdict", "unknown")
        pred = verdict_to_bool(raw)

        if raw.lower() == "unrelated":
            unrelated_count += 1

        if pred is None:
            parse_err += 1
            continue

        if gt and pred:         tp += 1
        elif gt and not pred:   fn += 1
        elif not gt and pred:   fp += 1
        else:                   tn += 1

    total = tp + fp + tn + fn
    acc  = (tp + tn) / total if total else 0
    prec = tp / (tp + fp)    if (tp + fp) else 0
    rec  = tp / (tp + fn)    if (tp + fn) else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

    # Breakdown by ground-truth class
    true_rows  = [r for r in rows if r["ground_truth"] is True]
    false_rows = [r for r in rows if r["ground_truth"] is False]

    def acc_subset(subset):
        if not subset:
            return 0.0
        correct = sum(
            1 for r in subset
            if verdict_to_bool(r.get(f"{key}_verdict", "unknown")) == r["ground_truth"]
               and verdict_to_bool(r.get(f"{key}_verdict", "unknown")) is not None
        )
        return correct / len(subset)

    return {
        "accuracy":         acc,
        "precision":        prec,
        "recall":           rec,
        "f1":               f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "parse_errors":     parse_err,
        "unrelated_count":  unrelated_count,
        "acc_true_claims":  acc_subset(true_rows),
        "acc_false_claims": acc_subset(false_rows),
        "avg_latency_s":    _avg(rows, f"{key}_latency_s"),
        "avg_tokens":       _avg(rows, f"{key}_tokens"),
        "avg_api_calls":    _avg(rows, f"{key}_api_calls"),
    }


def _avg(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return round(sum(vals) / len(vals), 2) if vals else 0.0


# ── Data loading ───────────────────────────────────────────────────────────────

def load_csv(path: str, n: int) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("chart_img") and row.get("claim") and row.get("label"):
                rows.append(row)
    if len(rows) > n:
        step = len(rows) // n
        rows = [rows[i * step] for i in range(n)]
    return rows


# ── Main ───────────────────────────────────────────────────────────────────────

def main(n: int, output: str, systems: list[str]) -> None:
    csv_path = Path(__file__).parent / "data" / "chartcheck" / "test.csv"
    rows = load_csv(str(csv_path), n)

    # Each system expands to display columns: (entry_key_prefix, header_label)
    DISPLAY_COL_MAP = [
        ("visual_only",     [("visual_only",  "Visual")]),
        ("structured_only", [("structured_only", "Struct")]),
        ("weighted",        [("weighted_a2", "Agent2"), ("weighted_a3", "Agent3"), ("weighted", "Weighted")]),
        ("pipeline",        [("pipeline_a2", "Pipe-A2"), ("pipeline_a3", "Pipe-A3"), ("pipeline", "Pipeline")]),
    ]
    SUMMARY_ROW_MAP = [
        ("visual_only",     [("visual_only",  "Visual only (A2)")]),
        ("structured_only", [("structured_only", "Structured only (A3)")]),
        ("weighted",        [
            ("weighted_a2", "  Agent 2 (visual)"),
            ("weighted_a3", "  Agent 3 (struct)"),
            ("weighted",    "Weighted vote (A2+A3)"),
        ]),
        ("pipeline",        [
            ("pipeline_a2", "  Pipe-A2 (visual)"),
            ("pipeline_a3", "  Pipe-A3 (struct)"),
            ("pipeline",    "Pipeline (A2+A3+A4)"),
        ]),
    ]
    active_display = [
        (prefix, label)
        for sys, cols in DISPLAY_COL_MAP if sys in systems
        for prefix, label in cols
    ]

    print(f"\nModel   : {MODEL}")
    print(f"Cases   : {len(rows)}")
    print(f"Systems : {', '.join(systems)}")
    print(f"Output  : {output}\n")
    header_cols = "".join(f"{label+'(conf)':<18}" for _, label in active_display)
    print(f"{'#':<4} {'GT':<6} {header_cols}Claim")
    print("-" * 110)

    results = []
    pipeline_agree_count = 0
    pipeline_agree_correct = 0
    weighted_tiebreak_count = 0

    for idx, row in enumerate(rows, 1):
        claim = row["claim"].strip()
        gt    = row["label"].strip().upper() == "TRUE"
        short = claim[:45] + "..." if len(claim) > 45 else claim

        img_result = download_image(row["chart_img"].strip())
        if img_result is None:
            skip_cols = "".join(f"{'SKIP':<18}" for _ in active_display)
            print(f"{idx:<4} {'TRUE' if gt else 'FALSE':<6} {skip_cols}{short}")
            continue

        img_path, mime = img_result
        img_b64 = encode_b64(img_path)

        entry: dict = {"idx": idx, "claim": claim, "chart_img": row["chart_img"].strip(), "ground_truth": gt}

        if "visual_only" in systems:
            try:
                out = run_visual_only(claim, img_b64, mime)
                for k, v in out.items():
                    entry[f"visual_only_{k}"] = v
            except Exception as e:
                print(f"    [visual_only error] {e}")
                entry["visual_only_verdict"] = "error"
                entry["visual_only_latency_s"] = 0.0
                entry["visual_only_tokens"] = 0
                entry["visual_only_api_calls"] = 0

        if "structured_only" in systems:
            try:
                out = run_structured_only(claim, img_b64, mime)
                for k, v in out.items():
                    entry[f"structured_only_{k}"] = v
            except Exception as e:
                print(f"    [structured_only error] {e}")
                entry["structured_only_verdict"] = "error"
                entry["structured_only_latency_s"] = 0.0
                entry["structured_only_tokens"] = 0
                entry["structured_only_api_calls"] = 0

        if "weighted" in systems:
            try:
                out = run_weighted(claim, img_b64, mime)
                for k, v in out.items():
                    entry[f"weighted_{k}"] = v
            except Exception as e:
                print(f"    [weighted error] {e}")
                entry["weighted_verdict"] = "error"
                entry["weighted_a2_verdict"] = "error"
                entry["weighted_a3_verdict"] = "error"
                entry["weighted_latency_s"] = 0.0
                entry["weighted_tokens"] = 0
                entry["weighted_api_calls"] = 0

        if "pipeline" in systems:
            try:
                out = run_pipeline(claim, img_b64, mime)
                for k, v in out.items():
                    entry[f"pipeline_{k}"] = v
                p_a2_c = float(entry.get("pipeline_a2_confidence", 0.0))
                p_a3_c = float(entry.get("pipeline_a3_confidence", 0.0))
                entry["pipeline_confidence"] = round((p_a2_c + p_a3_c) / 2, 4)
            except Exception as e:
                print(f"    [pipeline error] {e}")
                entry["pipeline_verdict"] = "error"
                entry["pipeline_latency_s"] = 0.0
                entry["pipeline_tokens"] = 0
                entry["pipeline_api_calls"] = 0

        # Track weighted tiebreaks
        if entry.get("weighted_tiebreak"):
            weighted_tiebreak_count += 1

        # Track A2/A3 agreement for pipeline
        pa2_v = entry.get("pipeline_a2_verdict", "")
        pa3_v = entry.get("pipeline_a3_verdict", "")
        if pa2_v and pa3_v and pa2_v not in ("unknown", "error") and pa3_v not in ("unknown", "error"):
            if pa2_v == pa3_v:
                pipeline_agree_count += 1
                if verdict_to_bool(pa2_v) == gt:
                    pipeline_agree_correct += 1

        results.append(entry)

        def fmt(prefix: str) -> str:
            v = entry.get(f"{prefix}_verdict", "?")
            c = entry.get(f"{prefix}_confidence")
            pred = verdict_to_bool(v)
            match = pred == gt if pred is not None else None
            mark = "Y" if match else ("N" if match is False else "-")
            tb = "(TB)" if prefix == "weighted" and entry.get("weighted_tiebreak") else ""
            if c is not None:
                return f"{v[:8]} {mark} {float(c):.2f}{tb}"
            return f"{v[:9]} {mark}{tb}"

        row_str = (
            f"{idx:<4} {'TRUE' if gt else 'FALSE':<6} "
            + "".join(f"{fmt(prefix):<18}" for prefix, _ in active_display)
            + short
        )
        print(row_str)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)

    active_summary = [
        (key, label)
        for sys, rows_def in SUMMARY_ROW_MAP if sys in systems
        for key, label in rows_def
    ]

    print(f"\n{'System':<26} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}  "
          f"{'Acc TRUE':>9} {'Acc FALSE':>10}  {'Unrel':>6} {'Err':>5}  "
          f"{'Latency':>8} {'Tokens':>7} {'Calls':>6}")
    print("-" * 102)

    all_metrics = {}
    for key, label in active_summary:
        m = compute_metrics(results, key)
        all_metrics[key] = m
        latency_str = f"{m['avg_latency_s']:>7.1f}s" if m["avg_latency_s"] > 0 else "       -"
        tokens_str  = f"{m['avg_tokens']:>7.0f}"      if m["avg_tokens"]   > 0 else "      -"
        calls_str   = f"{m['avg_api_calls']:>6.1f}"   if m["avg_api_calls"] > 0 else "     -"
        print(
            f"{label:<26} {m['accuracy']:>6.3f} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}  "
            f"{m['acc_true_claims']:>9.3f} {m['acc_false_claims']:>10.3f}  "
            f"{m['unrelated_count']:>6} {m['parse_errors']:>5}  "
            f"{latency_str} {tokens_str} {calls_str}"
        )

    print("-" * 102)

    # Weighted tiebreak summary
    if "weighted" in systems and weighted_tiebreak_count > 0:
        print(f"\nWeighted Agent 4 tiebreaks: {weighted_tiebreak_count}/{len(results)} cases "
              f"({weighted_tiebreak_count/len(results):.0%}) escalated to Opus 4.7")

    # A2 vs A3 agreement analysis (only if pipeline was run)
    if "pipeline" in systems and pipeline_agree_count > 0:
        agree_acc = pipeline_agree_correct / pipeline_agree_count
        total_pipeline = len(results)
        print(f"\nPipeline A2/A3 agreement: {pipeline_agree_count}/{total_pipeline} cases agree "
              f"({pipeline_agree_count/total_pipeline:.0%}), "
              f"accuracy when both agree: {agree_acc:.3f}")

    # Best system (top-level only, not sub-rows)
    top_keys = {
        "visual_only":     "Visual only (A2)",
        "structured_only": "Structured only (A3)",
        "weighted":        "Weighted vote (A2+A3)",
        "pipeline":        "Pipeline (A2+A3+A4)",
    }
    top_metrics = {k: all_metrics[k] for k in top_keys if k in all_metrics}
    if top_metrics:
        best = max(top_metrics, key=lambda k: top_metrics[k]["f1"])
        print(f"\nBest F1: {top_keys[best]}  ({top_metrics[best]['f1']:.3f})")
    print("=" * 100)

    # ── Save CSV ───────────────────────────────────────────────────────────────
    if results:
        fieldnames = list(results[0].keys())
        with open(output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nDetailed results saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fact-checking systems")
    parser.add_argument("--n",      type=int, default=20,              help="Test cases to evaluate (default: 20)")
    parser.add_argument("--output", type=str, default="eval_results.csv", help="Output CSV path")
    parser.add_argument(
        "--systems", type=str,
        default="visual_only,structured_only,weighted,pipeline",
        help="Comma-separated systems to report: visual_only,structured_only,weighted,pipeline (default: all)"
    )
    args = parser.parse_args()
    valid = {"visual_only", "structured_only", "weighted", "pipeline"}
    chosen = [s.strip() for s in args.systems.split(",") if s.strip() in valid]
    if not chosen:
        parser.error(f"--systems must include at least one of: {', '.join(sorted(valid))}")
    main(args.n, args.output, chosen)
