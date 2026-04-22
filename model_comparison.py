"""
model_comparison.py — Find the best model for each agent task.

Tests all 6 agents with every model in MODELS and ranks them by task-specific
metrics. Agents 3, 4, 6 use ChartFC accuracy; Agents 1, 2, 5 use quality
proxies (JSON validity, format correctness, latency).

== CONFIGURE ==============================================================
  MODELS      — OpenRouter model IDs to compare
  AGENTS      — which agents to include (default: all 6)
  LIMIT       — ChartFC samples per accuracy run (20 = fast, 100 = solid)
  SPLIT       — "test" for final numbers, "val" while tuning prompts
  JUDGE_BASE  — model used for agents 3 & 4 when isolating the judge (Agent 6)
  CONCURRENCY — parallel LLM calls (increase carefully to avoid rate limits)

== RUN ====================================================================
  python model_comparison.py

Results -> output/comparison_<timestamp>/
"""

import asyncio
import json
import re
import time
from datetime import datetime
from pathlib import Path

import litellm
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / "chart_verifier" / ".env")

# ==========================================================================
# CONFIGURE — edit before running
# ==========================================================================

MODELS = [
    "openrouter/openai/gpt-4o",
    "openrouter/anthropic/claude-3.5-haiku",
    "openrouter/openai/gpt-4o-mini",
]

# Which agents to benchmark. Remove any you don't need.
#   accuracy-based  (ChartFC TRUE/FALSE): visual, structured, judge
#   quality-based   (format + latency):   decomposer, typing, verification
AGENTS = [
    "decomposer",    # Agent 1 — claim decomposer
    "typing",        # Agent 2 — claim typing
    "visual",        # Agent 3 — visual evidence          (needs vision model)
    "structured",    # Agent 4 — structured evidence      (needs vision model)
    "verification",  # Agent 5 — verification question
    "judge",         # Agent 6 — judge & feedback
]

LIMIT       = 5       # ChartFC samples per accuracy run
SPLIT       = "test"  # "test" or "val"
JUDGE_BASE  = MODELS[0]   # fixed model for evidence (agents 3&4) when testing agent 6
CONCURRENCY = 3           # max parallel LLM calls per batch

OUTPUT_DIR = Path(__file__).parent / "output"

# ==========================================================================

from benchmark_eval import (
    load_samples, download_image, encode_image, remove_temp,
    call_llm, _typed_claims_json, _image_parts, _parse_first_claim,
    run_visual, run_structured, run_judge,
    candidate_to_verdict, feedback_to_verdict, compute_metrics,
)
from chart_verifier.agents.claim_decomposer   import INSTRUCTION as DECOMPOSER_INSTRUCTION
from chart_verifier.agents.claim_typing       import INSTRUCTION as TYPING_INSTRUCTION
from chart_verifier.agents.visual_evidence    import INSTRUCTION as VISUAL_INSTRUCTION
from chart_verifier.agents.structured_evidence import INSTRUCTION as STRUCTURED_INSTRUCTION
from chart_verifier.agents.verification_question import INSTRUCTION as VERIFICATION_INSTRUCTION
from chart_verifier.agents.judge_feedback     import INSTRUCTION as JUDGE_INSTRUCTION
from chart_verifier.agent import _extract_json_array


# -- Valid values for claim_typing output --------------------------------------
VALID_CLAIM_TYPES = {
    "numeric_lookup", "trend", "comparison", "ranking",
    "composition_share", "unsupported_interpretation", "unrelated",
}
VERIFIABLE_TYPES = VALID_CLAIM_TYPES - {"unrelated", "unsupported_interpretation"}
VALID_VERDICTS = {
    "supported", "contradicted", "partially_supported",
    "insufficient_evidence", "unrelated",
}


def slugify(model: str) -> str:
    name = model.split("/")[-1]
    return re.sub(r"[^\w\-]", "_", name)


# -- Agent 1: Claim Decomposer -------------------------------------------------

async def evaluate_decomposer(model: str, samples: list[dict]) -> dict:
    """
    Quality metrics:
      json_valid_rate  — fraction of calls that returned a parseable JSON array
      avg_claims       — average number of claims extracted per narrative
      claim_nonempty   — fraction where at least one claim was extracted
      avg_latency_s    — seconds per call
    """
    rows = []
    sem = asyncio.Semaphore(CONCURRENCY)

    async def _eval_one(row: dict) -> dict:
        claim = row["claim"].strip()
        user_text = (
            f"Claims from the Claim Decomposer Agent:\n"
            f"Student narrative: {claim}"
        )
        t0 = time.perf_counter()
        try:
            async with sem:
                raw = await call_llm(model, DECOMPOSER_INSTRUCTION,
                                     [{"type": "text", "text": user_text}])
        except Exception as exc:
            print(f"    Error: {exc}")
            return {"claim": claim, "json_valid": False, "n_claims": 0,
                    "nonempty": False, "latency_s": time.perf_counter() - t0,
                    "raw_response": f"ERROR: {exc}"}
        latency = time.perf_counter() - t0

        arr = _extract_json_array(raw)
        json_valid = arr is not None
        n_claims   = len(arr) if arr else 0
        nonempty   = json_valid and n_claims > 0 and all(
            isinstance(c, str) and c.strip() for c in arr
        )
        return {
            "claim": claim,
            "json_valid": json_valid,
            "n_claims": n_claims,
            "nonempty": nonempty,
            "latency_s": latency,
            "raw_response": raw[:300],
        }

    tasks = [_eval_one(s) for s in samples]
    print(f"  Running {len(tasks)} decomposer evaluations...")
    rows = await asyncio.gather(*tasks)

    total = len(rows)
    json_valid_rate = sum(r["json_valid"] for r in rows) / total
    avg_claims      = sum(r["n_claims"]   for r in rows) / total
    nonempty_rate   = sum(r["nonempty"]   for r in rows) / total
    avg_latency     = sum(r["latency_s"]  for r in rows) / total

    # Score: weight json validity heavily, reward reasonable claim count
    score = json_valid_rate * 0.5 + nonempty_rate * 0.3 + min(avg_claims / 2.0, 1.0) * 0.2

    return {
        "agent": "decomposer",
        "model": model,
        "metrics": {
            "score":           round(score, 3),
            "json_valid_rate": round(json_valid_rate, 3),
            "nonempty_rate":   round(nonempty_rate, 3),
            "avg_claims":      round(avg_claims, 2),
            "avg_latency_s":   round(avg_latency, 2),
            "total":           total,
        },
        "rows": list(rows),
    }


# -- Agent 2: Claim Typing -----------------------------------------------------

async def evaluate_typing(model: str, samples: list[dict]) -> dict:
    """
    Quality metrics:
      json_valid_rate        — parseable JSON array of typed claims
      valid_type_rate        — all claim_type values are in VALID_CLAIM_TYPES
      verifiable_rate        — fraction typed as a verifiable type (not unrelated)
      required_fields_rate   — all required fields present (claim, type, short_circuit)
      avg_latency_s
    """
    rows = []
    sem = asyncio.Semaphore(CONCURRENCY)

    async def _eval_one(row: dict) -> dict:
        claim = row["claim"].strip()
        # The typing agent expects output from decomposer — feed the claim as if already decomposed
        decomposed_json = json.dumps([claim])
        user_text = f"Claims from the Claim Decomposer Agent:\n{decomposed_json}"

        t0 = time.perf_counter()
        try:
            async with sem:
                raw = await call_llm(model, TYPING_INSTRUCTION,
                                     [{"type": "text", "text": user_text}])
        except Exception as exc:
            print(f"    Error: {exc}")
            return {"claim": claim, "json_valid": False, "valid_types": False,
                    "verifiable": False, "required_fields": False,
                    "latency_s": time.perf_counter() - t0, "raw_response": f"ERROR: {exc}"}
        latency = time.perf_counter() - t0

        arr = _extract_json_array(raw)
        json_valid = arr is not None and len(arr) > 0

        valid_types     = False
        verifiable      = False
        required_fields = False

        if json_valid:
            item = arr[0] if isinstance(arr[0], dict) else {}
            claim_type = item.get("type", "")
            valid_types     = claim_type in VALID_CLAIM_TYPES
            verifiable      = claim_type in VERIFIABLE_TYPES
            required_fields = all(k in item for k in ("claim", "type", "short_circuit"))

        return {
            "claim":           claim,
            "json_valid":      json_valid,
            "valid_types":     valid_types,
            "verifiable":      verifiable,
            "required_fields": required_fields,
            "latency_s":       latency,
            "raw_response":    raw[:300],
        }

    tasks = [_eval_one(s) for s in samples]
    print(f"  Running {len(tasks)} typing evaluations...")
    rows = await asyncio.gather(*tasks)

    total               = len(rows)
    json_valid_rate     = sum(r["json_valid"]      for r in rows) / total
    valid_type_rate     = sum(r["valid_types"]     for r in rows) / total
    verifiable_rate     = sum(r["verifiable"]      for r in rows) / total
    req_fields_rate     = sum(r["required_fields"] for r in rows) / total
    avg_latency         = sum(r["latency_s"]       for r in rows) / total

    score = json_valid_rate * 0.3 + valid_type_rate * 0.3 + req_fields_rate * 0.25 + verifiable_rate * 0.15

    return {
        "agent": "typing",
        "model": model,
        "metrics": {
            "score":              round(score, 3),
            "json_valid_rate":    round(json_valid_rate, 3),
            "valid_type_rate":    round(valid_type_rate, 3),
            "verifiable_rate":    round(verifiable_rate, 3),
            "req_fields_rate":    round(req_fields_rate, 3),
            "avg_latency_s":      round(avg_latency, 2),
            "total":              total,
        },
        "rows": list(rows),
    }


# -- Agent 5: Verification Question -------------------------------------------

async def evaluate_verification(model: str, samples: list[dict]) -> dict:
    """
    Quality metrics (no ground truth — uses format + logic checks):
      json_valid_rate        — parseable JSON array
      required_fields_rate   — claim, needed, verification_question, answer_from_evidence
      question_when_needed   — when needed=true, a question was actually produced
      valid_verdict_rate     — resolved_verdict in allowed values (or null when not needed)
      avg_latency_s
    """
    rows = []
    sem = asyncio.Semaphore(CONCURRENCY)

    async def _eval_one(row: dict) -> dict:
        claim = row["claim"].strip()

        typed_claims = json.dumps([{
            "claim": claim,
            "type": "comparison",
            "reasoning": "Benchmark test",
            "short_circuit": False,
            "short_circuit_verdict": None,
        }])
        # Simulate low-confidence evidence to trigger needed=true
        visual_ev = json.dumps([{
            "claim": claim, "skipped": False,
            "evidence_summary": "The chart shows some data, but values are hard to read.",
            "candidate_answer": "partially_supported",
            "confidence": 0.45,
        }])
        struct_ev = json.dumps([{
            "claim": claim, "skipped": False,
            "extracted_table": "Category | Value\nA | ~10\nB | ~12",
            "candidate_answer": "insufficient_evidence",
            "confidence": 0.40,
        }])
        context = (
            f"Typed claims from the Claim Typing Agent:\n{typed_claims}\n\n"
            f"Visual evidence from the Visual Evidence Agent:\n{visual_ev}\n\n"
            f"Structured evidence from the Structured Evidence Agent:\n{struct_ev}"
        )

        t0 = time.perf_counter()
        try:
            async with sem:
                raw = await call_llm(model, VERIFICATION_INSTRUCTION,
                                     [{"type": "text", "text": context}])
        except Exception as exc:
            print(f"    Error: {exc}")
            return {"claim": claim, "json_valid": False, "required_fields": False,
                    "question_when_needed": False, "valid_verdict": False,
                    "latency_s": time.perf_counter() - t0, "raw_response": f"ERROR: {exc}"}
        latency = time.perf_counter() - t0

        arr = _extract_json_array(raw)
        json_valid = arr is not None and len(arr) > 0

        required_fields    = False
        question_when_needed = True   # vacuously true if needed=false
        valid_verdict      = False

        if json_valid and isinstance(arr[0], dict):
            item = arr[0]
            required_fields = all(k in item for k in
                                  ("claim", "needed", "verification_question", "answer_from_evidence"))
            needed = item.get("needed", False)
            if needed:
                question_when_needed = bool(item.get("verification_question"))
            verdict = item.get("resolved_verdict")
            valid_verdict = (verdict is None and not needed) or verdict in VALID_VERDICTS

        return {
            "claim":                  claim,
            "json_valid":             json_valid,
            "required_fields":        required_fields,
            "question_when_needed":   question_when_needed,
            "valid_verdict":          valid_verdict,
            "latency_s":              latency,
            "raw_response":           raw[:300],
        }

    tasks = [_eval_one(s) for s in samples]
    print(f"  Running {len(tasks)} verification evaluations...")
    rows = await asyncio.gather(*tasks)

    total                = len(rows)
    json_valid_rate      = sum(r["json_valid"]           for r in rows) / total
    req_fields_rate      = sum(r["required_fields"]      for r in rows) / total
    q_when_needed_rate   = sum(r["question_when_needed"] for r in rows) / total
    valid_verdict_rate   = sum(r["valid_verdict"]        for r in rows) / total
    avg_latency          = sum(r["latency_s"]            for r in rows) / total

    score = (json_valid_rate * 0.30 + req_fields_rate * 0.30 +
             q_when_needed_rate * 0.25 + valid_verdict_rate * 0.15)

    return {
        "agent": "verification",
        "model": model,
        "metrics": {
            "score":               round(score, 3),
            "json_valid_rate":     round(json_valid_rate, 3),
            "req_fields_rate":     round(req_fields_rate, 3),
            "q_when_needed_rate":  round(q_when_needed_rate, 3),
            "valid_verdict_rate":  round(valid_verdict_rate, 3),
            "avg_latency_s":       round(avg_latency, 2),
            "total":               total,
        },
        "rows": list(rows),
    }


# -- Agents 3, 4, 6 — accuracy-based (ChartFC) --------------------------------

async def evaluate_accuracy_agent(
    agent: str,
    model: str,
    samples: list[dict],
    base_model: str,
) -> dict:
    """
    Accuracy-based evaluation using ChartFC TRUE/FALSE ground truth.
    Wraps the logic from benchmark_eval.evaluate() for a single agent/model.
    """
    rows: list[dict] = []
    sem = asyncio.Semaphore(CONCURRENCY)

    async def _eval_one(row: dict) -> dict:
        claim        = row["claim"].strip()
        ground_truth = row["label"].strip().upper()
        img_url      = row["chart_img"].strip()
        predicted    = "UNKNOWN"
        extra: dict  = {}

        async with sem:
            img_path, mime = download_image(img_url)
            if not img_path:
                return {"ground_truth": ground_truth, "predicted": "UNKNOWN",
                        "claim": claim}
            try:
                img_b64 = encode_image(img_path)
            finally:
                remove_temp(img_path)

            t0 = time.perf_counter()
            try:
                if agent == "visual":
                    result = await run_visual(claim, img_b64, mime, model)
                    if result:
                        predicted = candidate_to_verdict(result.get("candidate_answer"))
                        extra = {"confidence": result.get("confidence"),
                                 "candidate_answer": result.get("candidate_answer")}

                elif agent == "structured":
                    result = await run_structured(claim, img_b64, mime, model)
                    if result:
                        predicted = candidate_to_verdict(result.get("candidate_answer"))
                        extra = {"confidence": result.get("confidence"),
                                 "candidate_answer": result.get("candidate_answer")}

                elif agent == "judge":
                    vis  = await run_visual(claim, img_b64, mime, base_model)
                    strc = await run_structured(claim, img_b64, mime, base_model)
                    if vis and strc:
                        feedback  = await run_judge(claim, vis, strc, model)
                        predicted = feedback_to_verdict(feedback)
                        extra     = {"feedback": feedback[:200]}

            except Exception as exc:
                print(f"    Error on '{claim[:50]}': {exc}")

            latency = time.perf_counter() - t0

        match = "OK" if predicted == ground_truth else "XX"
        print(f"    {match} GT={ground_truth} Pred={predicted}  {claim[:55]}")
        return {"ground_truth": ground_truth, "predicted": predicted,
                "claim": claim, "latency_s": latency, **extra}

    tasks = [_eval_one(s) for s in samples]
    print(f"  Running {len(tasks)} {agent} evaluations...")
    rows = await asyncio.gather(*tasks)

    metrics = compute_metrics(list(rows))
    metrics["avg_latency_s"] = round(
        sum(r.get("latency_s", 0) for r in rows) / len(rows), 2
    )
    # Unified score for ranking: F1 (primary) with accuracy as tiebreaker
    metrics["score"] = round(metrics["f1"] * 0.7 + metrics["accuracy"] * 0.3, 3)

    return {
        "agent":      agent,
        "model":      model,
        "base_model": base_model,
        "metrics":    metrics,
        "rows":       list(rows),
    }


# -- Dispatcher ----------------------------------------------------------------

async def run_agent_model(
    agent: str,
    model: str,
    samples: list[dict],
    base_model: str,
) -> dict:
    """Route to the right evaluator for this agent."""
    print(f"\n  [{agent.upper()}] model={slugify(model)}")

    if agent == "decomposer":
        return await evaluate_decomposer(model, samples)
    elif agent == "typing":
        return await evaluate_typing(model, samples)
    elif agent == "verification":
        return await evaluate_verification(model, samples)
    elif agent in ("visual", "structured", "judge"):
        return await evaluate_accuracy_agent(agent, model, samples, base_model)
    else:
        raise ValueError(f"Unknown agent: {agent!r}")


# -- Save & display ------------------------------------------------------------

METRIC_LABELS = {
    "decomposer":    ("score", "json_valid_rate", "nonempty_rate", "avg_claims", "avg_latency_s"),
    "typing":        ("score", "json_valid_rate", "valid_type_rate", "verifiable_rate", "req_fields_rate", "avg_latency_s"),
    "visual":        ("score", "accuracy", "f1", "precision", "recall", "avg_latency_s"),
    "structured":    ("score", "accuracy", "f1", "precision", "recall", "avg_latency_s"),
    "verification":  ("score", "json_valid_rate", "req_fields_rate", "q_when_needed_rate", "valid_verdict_rate", "avg_latency_s"),
    "judge":         ("score", "accuracy", "f1", "precision", "recall", "avg_latency_s"),
}

AGENT_DESCRIPTIONS = {
    "decomposer":   "Agent 1 — Claim Decomposer   (quality proxy)",
    "typing":       "Agent 2 — Claim Typing        (quality proxy)",
    "visual":       "Agent 3 — Visual Evidence     (ChartFC accuracy)",
    "structured":   "Agent 4 — Structured Evidence (ChartFC accuracy)",
    "verification": "Agent 5 — Verification Q      (quality proxy)",
    "judge":        "Agent 6 — Judge & Feedback    (ChartFC accuracy)",
}


def print_summary(all_results: list[dict]) -> None:
    by_agent: dict[str, list] = {}
    for r in all_results:
        by_agent.setdefault(r["agent"], []).append(r)

    print(f"\n{'='*76}")
    print("  MODEL COMPARISON SUMMARY")
    print(f"{'='*76}")

    best_per_agent: dict[str, str] = {}

    for agent in AGENTS:
        group = sorted(
            by_agent.get(agent, []),
            key=lambda x: x["metrics"].get("score", 0),
            reverse=True,
        )
        if not group:
            continue

        desc = AGENT_DESCRIPTIONS.get(agent, agent)
        print(f"\n  {desc}")

        cols = METRIC_LABELS.get(agent, ("score",))
        header = f"  {'Model':<36}" + "".join(f"  {c[:10]:>10}" for c in cols)
        print(header)
        print("  " + "-" * (len(header) - 2))

        for i, r in enumerate(group):
            slug = slugify(r["model"])
            star = " *" if i == 0 else "  "
            vals = "".join(
                f"  {r['metrics'].get(c, 0):>10.3f}" for c in cols
            )
            print(f"  {slug:<36}{vals}{star}")

        best_per_agent[agent] = group[0]["model"]

    print(f"\n{'='*76}")
    print("  RECOMMENDED MODELS PER AGENT")
    print(f"{'='*76}")
    for agent in AGENTS:
        best = best_per_agent.get(agent, "N/A")
        print(f"  {agent:<14} -> {best}")
    print(f"{'='*76}\n")


# -- Main ----------------------------------------------------------------------

async def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR / f"comparison_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*76}")
    print("  Chart Verifier — Per-Agent Model Comparison")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Models  : {len(MODELS)} — {', '.join(slugify(m) for m in MODELS)}")
    print(f"  Agents  : {', '.join(AGENTS)}")
    print(f"  Samples : {LIMIT} (ChartFC {SPLIT} split)")
    print(f"  Output  : {out_dir}/")
    print(f"{'='*76}")

    print(f"\nLoading {LIMIT} samples from ChartFC {SPLIT} split...")
    samples = load_samples(SPLIT, LIMIT)
    print(f"Loaded {len(samples)} samples.\n")

    all_results: list[dict] = []
    total_runs = len(AGENTS) * len(MODELS)
    run_num    = 0

    for agent in AGENTS:
        print(f"\n{'='*76}")
        print(f"  AGENT: {AGENT_DESCRIPTIONS.get(agent, agent)}")
        print(f"{'='*76}")

        for model in MODELS:
            run_num += 1
            print(f"\n[{run_num}/{total_runs}]")

            base = JUDGE_BASE if agent == "judge" else model

            result = await run_agent_model(agent, model, samples, base)
            all_results.append(result)

            # Save individual result
            slug = slugify(model)
            result_path = out_dir / f"{agent}__{slug}.json"
            save_result = {k: v for k, v in result.items() if k != "rows"}
            save_result["rows"] = result.get("rows", [])
            with open(result_path, "w") as f:
                json.dump(save_result, f, indent=2)
            print(f"  Saved -> {result_path.name}")

            m = result["metrics"]
            score_val = m.get("score", 0)
            print(f"  Score={score_val:.3f}", end="")
            for k, v in m.items():
                if k != "score" and k != "total" and k != "rows":
                    print(f"  {k}={v}", end="")
            print()

    # Save combined summary
    summary = {
        "run_at":     datetime.now().isoformat(),
        "split":      SPLIT,
        "limit":      LIMIT,
        "judge_base": JUDGE_BASE,
        "models":     MODELS,
        "agents":     AGENTS,
        "results": [
            {"agent": r["agent"], "model": r["model"], "metrics": r["metrics"]}
            for r in all_results
        ],
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print_summary(all_results)
    print(f"Full results -> {out_dir}/")
    print(f"Summary      -> {summary_path}\n")


if __name__ == "__main__":
    asyncio.run(main())
