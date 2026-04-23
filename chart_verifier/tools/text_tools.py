"""
Deterministic text processing tools.
These do NOT need an LLM — they are fast, cheap, and reliable.
"""

import re


def split_sentences(text: str) -> dict:
    """
    Split a paragraph into individual sentences using rule-based splitting.
    No LLM needed — spaCy-style boundary detection via regex.

    Args:
        text: The student narrative paragraph to split.

    Returns:
        dict with 'sentences' (list of str) and 'count' (int).
    """
    # Normalize whitespace
    text = text.strip()

    # Split on sentence boundaries: period/!/?  followed by space + capital letter
    # or end of string. Handles abbreviations imperfectly but good enough for
    # student chart narratives which are typically short.
    raw = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    sentences = [s.strip() for s in raw if s.strip()]
    return {"sentences": sentences, "count": len(sentences)}



def format_verdict_summary(verdicts: list[dict]) -> dict:
    """
    Aggregate a list of per-claim verdicts into a summary count.
    Pure computation — no LLM needed.

    Args:
        verdicts: List of dicts, each with a 'verdict' key.
                  Expected verdict values: supported, contradicted,
                  partially_supported, insufficient_evidence, unrelated.

    Returns:
        dict with counts per verdict type and 'total' claim count.
    """
    counts = {
        "supported": 0,
        "contradicted": 0,
        "partially_supported": 0,
        "insufficient_evidence": 0,
        "unrelated": 0,
        "unknown": 0,
    }

    for v in verdicts:
        label = v.get("final_verdict", v.get("verdict", "unknown")).lower().replace(" ", "_")
        if label in counts:
            counts[label] += 1
        else:
            counts["unknown"] += 1

    counts["total"] = len(verdicts)
    return counts
