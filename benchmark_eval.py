"""
Benchmark individual agents or the full pipeline against the ChartCheck dataset.

ChartCheck: "Explainable Fact-Checking over Real-World Chart Images"
-- ACL Findings 2024.  Labels: TRUE / FALSE.
  Test t1 : 939 claims
  Test t2 : 981 claims

Two evaluation modes run together whenever the agent is 'visual' or 'structured':

  Classification metrics  — Accuracy / Precision / Recall / F1
  NLG metrics (ChartCheck style) — BLEU / ROUGE-L / METEOR / BERTScore
    Generated evidence_summary+reasoning is compared against the gold
    'explanation' column, mirroring the ChartCheck paper setup.

Paper baselines (ChartCheck test1):
  MatCha-finetune-class       : Accuracy 64.0%  F1 63.7%
  DePlot-FlanT5-finetune      : Accuracy 69.6%  F1 69.6%
  DePlot-DeBERTa (best)       : Accuracy 74.9%  F1 74.9%

== AGENT MODES ============================================================
  visual      Agent 3 -- reads the chart image visually  (needs a vision model)
  structured  Agent 4 -- extracts a data table, then reasons over it  (vision)
  judge       Agent 6 -- arbitrates evidence -> TRUE/FALSE verdict  (text model)
              Requires --base-model for agents 3 & 4 to generate evidence first
  full        All 6 agents end-to-end

== EXAMPLES ===============================================================

  python benchmark_eval.py --agent visual --model openrouter/anthropic/claude-3.5-sonnet --limit 30
  python benchmark_eval.py --agent structured --model openrouter/google/gemini-2.5-flash-preview --limit 30
  python benchmark_eval.py --agent judge --model openrouter/deepseek/deepseek-chat --base-model openrouter/anthropic/claude-3.5-sonnet --limit 30
  python benchmark_eval.py --compare data/results/visual_claude.json data/results/visual_gpt4o.json

===========================================================================
"""

import argparse
import asyncio
import base64
import csv
import json
import os
import tempfile
import urllib.request
from pathlib import Path

import litellm
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / "chart_verifier" / ".env")

# -- Paths ---------------------------------------------------------------------
DATA_DIR    = Path(__file__).parent / "data" / "chartcheck"
RESULTS_DIR = Path(__file__).parent / "data" / "results"

_BASE = "https://raw.githubusercontent.com/mubasharaak/ChartCheck/main/data"
SPLIT_URLS = {
    "test":  f"{_BASE}/claim_explanation_verification_pre_tasksets_test_V2.csv",
    "test2": f"{_BASE}/claim_explanation_verification_pre_tasksets_test_two_V2.csv",
    "val":   f"{_BASE}/claim_explanation_verification_pre_tasksets_validation_V2.csv",
    "train": f"{_BASE}/claim_explanation_verification_pre_tasksets_train_V2.csv",
}


# -- Agent instructions --------------------------------------------------------
from chart_verifier.agents.visual_evidence    import INSTRUCTION as VISUAL_INSTRUCTION
from chart_verifier.agents.structured_evidence import INSTRUCTION as STRUCTURED_INSTRUCTION
from chart_verifier.agents.judge_feedback      import INSTRUCTION as JUDGE_INSTRUCTION


# -- NLG metric imports --------------------------------------------------------
# Required packages: pip install rouge-score bert-score
# nltk data: nltk.download('punkt_tab'); nltk.download('wordnet')

try:
    import nltk
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer as _rouge_scorer_module
    from bert_score import score as _bert_score_fn
    _NLG_AVAILABLE = True
except ImportError:
    _NLG_AVAILABLE = False


# -- Data helpers --------------------------------------------------------------

def load_samples(split: str, limit: int | None) -> list[dict]:
    dest = DATA_DIR / f"{split}.csv"
    if not dest.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {split}.csv ...")
        urllib.request.urlretrieve(SPLIT_URLS[split], dest)

    samples = []
    with open(dest, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            samples.append(row)
            if limit and len(samples) >= limit:
                break
    return samples


def download_image(url: str) -> tuple[str | None, str]:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
            urllib.request.urlretrieve(url, tmp.name)
            path = tmp.name

        # Detect actual format from file magic bytes, not URL extension
        with open(path, "rb") as f:
            header = f.read(12)
        if header[:4] == b'\x89PNG':
            mime, ext = "image/png",  ".png"
        elif header[:3] in (b'\xff\xd8\xff',):
            mime, ext = "image/jpeg", ".jpg"
        elif header[:6] in (b'GIF87a', b'GIF89a'):
            mime, ext = "image/gif",  ".gif"
        elif header[:4] == b'RIFF' and header[8:12] == b'WEBP':
            mime, ext = "image/webp", ".webp"
        else:
            mime, ext = "image/png",  ".png"  # best guess

        new_path = path + ext
        os.rename(path, new_path)
        return new_path, mime
    except Exception as exc:
        print(f"    Image download failed: {exc}")
        return None, ""


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def remove_temp(path: str | None) -> None:
    if path and os.path.exists(path):
        os.unlink(path)


# -- LLM caller ----------------------------------------------------------------

async def call_llm(model: str, system: str, user_parts: list) -> str:
    response = await litellm.acompletion(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user_parts},
        ],
    )
    return response.choices[0].message.content or ""


# -- Simulated context helpers -------------------------------------------------

def _typed_claims_json(claim: str) -> str:
    return json.dumps([{
        "claim": claim,
        "type": "comparison",
        "reasoning": "Benchmark evaluation",
        "short_circuit": False,
        "short_circuit_verdict": None,
    }])


def _image_parts(img_b64: str, mime: str, extra_text: str) -> list:
    return [
        {"type": "text",      "text": extra_text},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
    ]


# -- Per-agent runners ---------------------------------------------------------

async def run_visual(claim: str, img_b64: str, mime: str, model: str) -> dict | None:
    context = f"Typed claims from Claim Typing Agent:\n{_typed_claims_json(claim)}"
    raw = await call_llm(model, VISUAL_INSTRUCTION, _image_parts(img_b64, mime, context))
    return _parse_first_claim(raw)


async def run_structured(claim: str, img_b64: str, mime: str, model: str) -> dict | None:
    context = f"Typed claims from Claim Typing Agent:\n{_typed_claims_json(claim)}"
    raw = await call_llm(model, STRUCTURED_INSTRUCTION, _image_parts(img_b64, mime, context))
    result = _parse_first_claim(raw)
    if result and result.get("verification_code"):
        from chart_verifier.tools.code_executor import execute_verification_code
        code_verdict, code_output = execute_verification_code(result["verification_code"])
        if code_verdict:
            result["candidate_answer"] = code_verdict
            result["code_executed"]    = True
            result["code_output"]      = code_output
        else:
            result["code_executed"] = False
            result["code_output"]   = code_output
    return result


async def run_judge(
    claim: str,
    visual_result: dict,
    structured_result: dict,
    model: str,
) -> str:
    typed   = _typed_claims_json(claim)
    visual  = json.dumps([visual_result])
    struct  = json.dumps([structured_result])
    verif   = json.dumps([{
        "claim": claim, "needed": False,
        "verification_question": None, "answer_from_evidence": None,
        "resolved_verdict": None, "confidence": None,
    }])
    context = (
        f"Typed claims (Claim Typing Agent):\n{typed}\n\n"
        f"Visual evidence (Visual Evidence Agent):\n{visual}\n\n"
        f"Structured evidence (Structured Evidence Agent):\n{struct}\n\n"
        f"Verification questions (Verification Question Agent):\n{verif}"
    )
    return await call_llm(model, JUDGE_INSTRUCTION,
                          [{"type": "text", "text": context}])


# -- Full pipeline runner -------------------------------------------------------

async def run_full_pipeline(claim: str, img_path: str, model: str) -> str:
    os.environ["OPENROUTER_MODEL"]        = model
    os.environ["OPENROUTER_VISION_MODEL"] = model
    from chart_verifier.orchestrator import run_pipeline
    return await run_pipeline(claim, img_path)


# -- Output parsers ------------------------------------------------------------

def _parse_first_claim(raw: str) -> dict | None:
    from chart_verifier.agent import _extract_json_array
    arr = _extract_json_array(raw)
    if arr and isinstance(arr[0], dict):
        return arr[0]
    return None


def candidate_to_verdict(answer: str | None) -> str:
    mapping = {
        "supported":             "TRUE",
        "contradicted":          "FALSE",
        "partially_supported":   "FALSE",
        "insufficient_evidence": "FALSE",
        "unrelated":             "FALSE",
        "true":                  "TRUE",
        "false":                 "FALSE",
        "yes":                   "TRUE",
        "no":                    "FALSE",
        "not supported":         "FALSE",
        "unsupported":           "FALSE",
    }
    normalised = (answer or "").lower().strip().replace(" ", "_")
    return mapping.get(normalised, "UNKNOWN")


def feedback_to_verdict(feedback: str) -> str:
    # Judge output has emoji section headers (✅/❌/⚠️/❓) followed by content or *(none)*.
    # We must skip sections whose next non-empty line contains "none" — those are empty sections.
    lines = feedback.splitlines()
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        verdict = None
        if s.startswith("✅"):                              verdict = "TRUE"
        elif s.startswith("❌"):                            verdict = "FALSE"
        elif s.startswith("⚠️"):                           verdict = "FALSE"
        elif s.startswith("❓") or s.startswith("🚫"):     verdict = "FALSE"

        if verdict:
            # Find next non-empty, non-separator line to check if section is "(none)"
            j = i + 1
            while j < len(lines) and (not lines[j].strip() or
                                       lines[j].strip().startswith("---")):
                j += 1
            if j < len(lines) and "none" not in lines[j].strip().lower():
                return verdict  # section has real content

        i += 1
    return "UNKNOWN"


def _evidence_text(result: dict | None) -> str:
    """Extract the agent's explanation text for NLG comparison."""
    if not result:
        return ""
    parts = []
    if result.get("evidence_summary"):
        parts.append(result["evidence_summary"])
    if result.get("reasoning"):
        parts.append(result["reasoning"])
    return " ".join(parts).strip()


# -- Classification metrics ----------------------------------------------------

def compute_metrics(rows: list[dict]) -> dict:
    tp = fp = tn = fn = unknown = 0
    for r in rows:
        gt, pred = r["ground_truth"], r["predicted"]
        if pred == "UNKNOWN":
            unknown += 1; continue
        if   gt == "TRUE"  and pred == "TRUE":  tp += 1
        elif gt == "TRUE"  and pred == "FALSE": fn += 1
        elif gt == "FALSE" and pred == "FALSE": tn += 1
        elif gt == "FALSE" and pred == "TRUE":  fp += 1
    scored    = tp + fp + tn + fn
    accuracy  = (tp + tn) / scored          if scored           else 0.0
    precision = tp / (tp + fp)              if (tp + fp)        else 0.0
    recall    = tp / (tp + fn)              if (tp + fn)        else 0.0
    f1        = 2*precision*recall / (precision+recall) if (precision+recall) else 0.0
    return dict(accuracy=accuracy, precision=precision, recall=recall, f1=f1,
                tp=tp, fp=fp, tn=tn, fn=fn, unknown=unknown,
                scored=scored, total=len(rows))


def print_metrics(label: str, m: dict) -> None:
    print(f"\n{'='*62}")
    print(f"RESULTS -- {label}")
    print(f"  Accuracy  : {m['accuracy']:.3f}")
    print(f"  Precision : {m['precision']:.3f}")
    print(f"  Recall    : {m['recall']:.3f}")
    print(f"  F1        : {m['f1']:.3f}")
    print(f"  TP={m['tp']}  FP={m['fp']}  TN={m['tn']}  FN={m['fn']}  "
          f"Unknown={m['unknown']}  Total={m['total']}")
    print("="*62)


# -- NLG metrics (ChartCheck-style) -------------------------------------------

def _tokenize_nlg(text: str) -> list[str]:
    return nltk.word_tokenize(text.lower())


def compute_nlg_metrics(hypotheses: list[str], references: list[str]) -> dict:
    """BLEU / ROUGE-L / METEOR / BERTScore over paired evidence vs gold explanation."""
    if not _NLG_AVAILABLE:
        print("  [NLG metrics skipped: run `pip install rouge-score bert-score`]")
        return {}
    if not hypotheses:
        return {}

    smooth   = SmoothingFunction().method1
    hyp_toks = [_tokenize_nlg(h) for h in hypotheses]
    ref_toks  = [[_tokenize_nlg(r)] for r in references]
    bleu = corpus_bleu(ref_toks, hyp_toks, smoothing_function=smooth) * 100

    scorer = _rouge_scorer_module.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = (sum(scorer.score(r, h)["rougeL"].fmeasure
                   for h, r in zip(hypotheses, references))
               / len(hypotheses) * 100)

    meteor = (sum(meteor_score([_tokenize_nlg(r)], _tokenize_nlg(h))
                  for h, r in zip(hypotheses, references))
              / len(hypotheses) * 100)

    print("  Computing BERTScore...")
    _, _, bert_f1 = _bert_score_fn(hypotheses, references, lang="en", verbose=False)
    bertscore = bert_f1.mean().item() * 100

    return dict(bleu=bleu, rouge_l=rouge_l, meteor=meteor, bertscore=bertscore,
                n_nlg=len(hypotheses))


def print_nlg_metrics(label: str, m: dict) -> None:
    if not m:
        return
    print(f"\n{'='*65}")
    print(f"ChartCheck NLG Metrics -- {label}  (n={m['n_nlg']})")
    print(f"  BLEU      : {m['bleu']:.1f}")
    print(f"  ROUGE-L   : {m['rouge_l']:.1f}")
    print(f"  METEOR    : {m['meteor']:.1f}")
    print(f"  BERTScore : {m['bertscore']:.1f}")
    print("="*65)


# -- Main evaluation loop -------------------------------------------------------

async def evaluate(
    agent:      str,
    model:      str,
    base_model: str,
    split:      str,
    limit:      int | None,
) -> dict:
    print(f"\n{'='*62}")
    print(f"Agent : {agent}")
    print(f"Model : {model}")
    if agent == "judge":
        print(f"Base  : {base_model}  (agents 3 & 4)")
    print(f"Split : {split}  |  Limit: {limit or 'all'}")
    print("="*62)

    samples = load_samples(split, limit)
    rows: list[dict] = []
    per_agent_rows: dict[str, list[dict]] = {"visual": [], "structured": [], "judge": []}

    # NLG tracking (visual / structured only — judge output isn't an explanation)
    nlg_hypotheses: list[str] = []
    nlg_references: list[str] = []

    for i, row in enumerate(samples, 1):
        claim        = row["claim"].strip()
        ground_truth = row["label"].strip().upper()
        img_url      = row["chart_img"].strip()
        gold_expl    = row.get("explanation", "").strip()

        print(f"\n[{i}/{len(samples)}] {claim[:70]}...")
        print(f"  Ground truth : {ground_truth}")

        predicted = "UNKNOWN"
        extra: dict = {}

        try:
            if agent in ("visual", "structured", "judge"):
                img_path, mime = download_image(img_url)
                if not img_path:
                    rows.append({"ground_truth": ground_truth, "predicted": "UNKNOWN",
                                 "claim": claim})
                    continue
                try:
                    img_b64 = encode_image(img_path)
                finally:
                    remove_temp(img_path)

                if agent == "visual":
                    result = await run_visual(claim, img_b64, mime, model)
                    if result:
                        predicted = candidate_to_verdict(result.get("candidate_answer"))
                        extra = {"confidence": result.get("confidence"),
                                 "candidate_answer": result.get("candidate_answer")}
                        hyp = _evidence_text(result)
                        if hyp and gold_expl:
                            nlg_hypotheses.append(hyp)
                            nlg_references.append(gold_expl)

                elif agent == "structured":
                    result = await run_structured(claim, img_b64, mime, model)
                    if result:
                        predicted = candidate_to_verdict(result.get("candidate_answer"))
                        extra = {"confidence": result.get("confidence"),
                                 "candidate_answer": result.get("candidate_answer")}
                        hyp = _evidence_text(result)
                        if hyp and gold_expl:
                            nlg_hypotheses.append(hyp)
                            nlg_references.append(gold_expl)

                elif agent == "judge":
                    vis  = await run_visual(claim, img_b64, mime, base_model)
                    strc = await run_structured(claim, img_b64, mime, base_model)
                    if vis and strc:
                        feedback  = await run_judge(claim, vis, strc, model)
                        predicted = feedback_to_verdict(feedback)
                        extra     = {"feedback": feedback[:200]}

            else:  # full pipeline
                img_path, mime = download_image(img_url)
                if not img_path:
                    skip = {"ground_truth": ground_truth, "predicted": "UNKNOWN", "claim": claim}
                    per_agent_rows["visual"].append(skip)
                    per_agent_rows["structured"].append(skip)
                    per_agent_rows["judge"].append(skip)
                    rows.append(skip)
                    continue
                try:
                    img_b64 = encode_image(img_path)
                finally:
                    remove_temp(img_path)

                vis_result  = await run_visual(claim, img_b64, mime, model)
                strc_result = await run_structured(claim, img_b64, mime, model)

                vis_pred  = candidate_to_verdict(vis_result.get("candidate_answer"))  if vis_result  else "UNKNOWN"
                strc_pred = candidate_to_verdict(strc_result.get("candidate_answer")) if strc_result else "UNKNOWN"

                judge_pred = "UNKNOWN"
                if vis_result and strc_result:
                    feedback   = await run_judge(claim, vis_result, strc_result, model)
                    judge_pred = feedback_to_verdict(feedback)
                    extra      = {"feedback": feedback[:200]}

                predicted = judge_pred

                per_agent_rows["visual"].append(
                    {"ground_truth": ground_truth, "predicted": vis_pred, "claim": claim,
                     "confidence": vis_result.get("confidence") if vis_result else None})
                per_agent_rows["structured"].append(
                    {"ground_truth": ground_truth, "predicted": strc_pred, "claim": claim,
                     "confidence": strc_result.get("confidence") if strc_result else None})
                per_agent_rows["judge"].append(
                    {"ground_truth": ground_truth, "predicted": judge_pred, "claim": claim})

                print(f"  Visual       : {vis_pred}  |  Structured: {strc_pred}  |  Judge: {judge_pred}")

        except Exception as exc:
            print(f"  Error: {exc}")

        match = "OK" if predicted == ground_truth else "XX"
        print(f"  Predicted    : {predicted}  {match}"
              + (f"  (confidence={extra.get('confidence', '?'):.2f})"
                 if "confidence" in extra and extra["confidence"] else ""))

        rows.append({"ground_truth": ground_truth, "predicted": predicted,
                     "claim": claim, **extra})

    metrics = compute_metrics(rows)
    print_metrics(f"{agent} / {model}", metrics)

    per_agent_metrics: dict = {}
    if agent == "full":
        for sub in ("visual", "structured", "judge"):
            sub_m = compute_metrics(per_agent_rows[sub])
            per_agent_metrics[sub] = sub_m
            print_metrics(f"full/{sub} / {model}", sub_m)

    # NLG metrics for visual / structured
    nlg_metrics: dict = {}
    if agent in ("visual", "structured") and nlg_hypotheses:
        nlg_metrics = compute_nlg_metrics(nlg_hypotheses, nlg_references)
        print_nlg_metrics(f"{agent} / {model}", nlg_metrics)

    return {"agent": agent, "model": model, "base_model": base_model,
            "split": split, "limit": limit, "metrics": metrics, "rows": rows,
            "per_agent_metrics": per_agent_metrics, "nlg_metrics": nlg_metrics}


# -- Compare mode --------------------------------------------------------------

def compare(paths: list[str]) -> None:
    records = []
    for p in paths:
        with open(p) as f:
            records.append(json.load(f))

    by_agent: dict[str, list] = {}
    for r in records:
        by_agent.setdefault(r["agent"], []).append(r)

    for agent, group in sorted(by_agent.items()):
        print(f"\n{'='*72}")
        print(f"AGENT: {agent}")

        # Classification
        print(f"\n  Classification:")
        print(f"  {'Model':<48} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
        print(f"  {'-'*72}")
        for r in sorted(group, key=lambda x: x["metrics"]["f1"], reverse=True):
            m = r["metrics"]
            print(f"  {r['model']:<48} {m['accuracy']:>6.3f} {m['precision']:>6.3f} "
                  f"{m['recall']:>6.3f} {m['f1']:>6.3f}")

        # NLG (visual / structured only)
        nlg_group = [r for r in group if r.get("nlg_metrics")]
        if nlg_group:
            print(f"\n  NLG (ChartCheck-style):")
            print(f"  {'Model':<48} {'BLEU':>5} {'ROUGE':>6} {'MET':>5} {'BERT':>6}")
            print(f"  {'-'*72}")
            for r in sorted(nlg_group, key=lambda x: x["nlg_metrics"].get("bertscore", 0), reverse=True):
                m = r["nlg_metrics"]
                print(f"  {r['model']:<48} {m['bleu']:>5.1f} {m['rouge_l']:>6.1f} "
                      f"{m['meteor']:>5.1f} {m['bertscore']:>6.1f}")

    print("="*72)


# -- CLI -----------------------------------------------------------------------

def main() -> None:
    default_model = os.getenv("OPENROUTER_MODEL",
                              "openrouter/anthropic/claude-3.5-sonnet")
    parser = argparse.ArgumentParser(
        description="Per-agent benchmark on ChartFC dataset (classification + ChartCheck NLG metrics)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--agent",      default="full",
                        choices=["visual", "structured", "judge", "full"])
    parser.add_argument("--model",      default=default_model)
    parser.add_argument("--base-model", default=default_model,
                        help="Model for agents 3&4 when --agent judge")
    parser.add_argument("--split",      default="test", choices=["test", "test2", "val", "train"])
    parser.add_argument("--limit",      type=int, default=20)
    parser.add_argument("--output",     default=None)
    parser.add_argument("--compare",    nargs="+", metavar="FILE")
    args = parser.parse_args()

    if args.compare:
        compare(args.compare)
        return

    base_model = args.base_model or args.model
    result = asyncio.run(evaluate(args.agent, args.model, base_model,
                                  args.split, args.limit))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved -> {out}")


if __name__ == "__main__":
    main()
