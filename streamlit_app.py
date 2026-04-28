"""
Streamlit UI for the Chart Verifier agent pipeline.

Run with:
    streamlit run streamlit_app.py
"""

import asyncio
import re
import tempfile
import os
import html
from difflib import SequenceMatcher

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chart Fact-Checker",
    page_icon="📊",
    layout="centered",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Hide Streamlit default chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Force dark page background */
.stApp { background-color: #0f1117 !important; }

/* Main container width */
.block-container { max-width: 780px; padding-top: 2.5rem; padding-bottom: 3rem; }

/* Page header */
.page-header {
    border-bottom: 2px solid #4f8ef7;
    padding-bottom: 0.75rem;
    margin-bottom: 1.75rem;
}
.page-header h1 {
    font-size: 1.75rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 0 0 0.25rem 0;
    letter-spacing: -0.3px;
}
.page-header p {
    font-size: 0.9rem;
    color: #94a3b8;
    margin: 0;
}

/* Section label */
.section-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 0.5rem;
}

/* Feedback box */
.feedback-box {
    background: #1a1d27;
    border: 1px solid #2d3148;
    border-left: 4px solid #4f8ef7;
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    font-size: 0.95rem;
    color: #e2e8f0;
    line-height: 1.75;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #2d3148;
    margin: 1.5rem 0;
}

/* Step list in status */
.step { font-size: 0.9rem; color: #cbd5e1; padding: 2px 0; }

/* Primary button override */
.stButton > button {
    background-color: #4f8ef7 !important;
    color: #0f1117 !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    padding: 0.45rem 1.5rem !important;
    font-size: 0.9rem !important;
}
.stButton > button:hover {
    background-color: #3a7be0 !important;
}
.stButton > button:disabled {
    background-color: #2d3148 !important;
    color: #64748b !important;
    cursor: not-allowed !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 1.5px dashed #2d3148 !important;
    border-radius: 8px !important;
    padding: 0.5rem !important;
}

/* Text area */
textarea {
    border-radius: 6px !important;
    border-color: #2d3148 !important;
    background-color: #1a1d27 !important;
    color: #e2e8f0 !important;
    font-size: 0.95rem !important;
}

/* Status box */
[data-testid="stStatus"] {
    border-radius: 8px !important;
    border-color: #2d3148 !important;
    background-color: #1a1d27 !important;
}

/* Highlighted narrative box */
.narrative-highlight-box {
    background: #1a1d27;
    border: 1px solid #2d3148;
    border-left: 4px solid #94a3b8;
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    font-size: 0.95rem;
    color: #e2e8f0;
    line-height: 2.0;
}

/* Verdict legend */
.verdict-legend {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-bottom: 0.75rem;
    font-size: 0.8rem;
}
.legend-item {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    color: #94a3b8;
}
.legend-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1>📊 Chart Fact-Checker</h1>
    <p>📤 Upload a chart and enter a student narrative. The pipeline will verify each claim against the chart data.</p>
</div>
""", unsafe_allow_html=True)


_VERDICT_STYLE = {
    "supported":    ("rgba(34,197,94,0.15)",  "#22c55e", "#86efac"),
    "contradicted": ("rgba(239,68,68,0.15)",  "#ef4444", "#fca5a5"),
    "unrelated":    ("rgba(234,179,8,0.15)",  "#eab308", "#fde047"),
}


def _word_similarity(a: str, b: str) -> float:
    wa = set(re.findall(r'\w+', a.lower()))
    wb = set(re.findall(r'\w+', b.lower()))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


_CAUSAL_RE = re.compile(
    r'\b(because|since|so|as|therefore|thus|hence|consequently)\b',
    re.IGNORECASE,
)


def _split_at_conjunction(sentence: str) -> list[str]:
    raw = _CAUSAL_RE.split(sentence)
    if len(raw) <= 1:
        return [sentence]
    fragments = []
    if raw[0].strip():
        fragments.append(raw[0].strip())
    i = 1
    while i + 1 <= len(raw) - 1:
        conj, text = raw[i], raw[i + 1].strip()
        if text:
            fragments.append(f"{conj} {text}")
        i += 2
    return fragments or [sentence]


def _make_span(text: str, verdict: str) -> str:
    bg, border, color = _VERDICT_STYLE[verdict]
    return (
        f'<span style="background:{bg};border-bottom:2px solid {border};'
        f'color:{color};padding:1px 3px;border-radius:3px;" '
        f'title="{verdict}">{html.escape(text)}</span>'
    )


def _best_verdict(text: str, claim_verdicts: list[dict], threshold: float) -> str | None:
    best_score, best_verdict = 0.0, None
    for cv in claim_verdicts:
        if cv["verdict"] not in _VERDICT_STYLE:
            continue
        score = _word_similarity(text, cv["claim"])
        if score >= threshold and score > best_score:
            best_score, best_verdict = score, cv["verdict"]
    return best_verdict


def highlight_narrative(narrative: str, claim_verdicts: list[dict]) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', narrative.strip())
    parts = []

    for sentence in sentences:
        sentence_lower = sentence.lower()

        # ── Primary: exact substring match on "original" field ────────────
        exact_matches = [
            cv for cv in claim_verdicts
            if cv["verdict"] in _VERDICT_STYLE
            and cv.get("original", "")
            and cv["original"].lower() in sentence_lower
        ]

        if exact_matches:
            if len(exact_matches) == 1 and exact_matches[0]["original"].lower() == sentence_lower:
                # Original covers the whole sentence — highlight entirely
                parts.append(_make_span(sentence, exact_matches[0]["verdict"]))
            else:
                # Multiple fragments or partial match — highlight each in place
                result = sentence
                for cv in sorted(exact_matches, key=lambda c: len(c["original"]), reverse=True):
                    original = cv["original"]
                    idx = result.lower().find(original.lower())
                    if idx != -1:
                        result = (
                            result[:idx]
                            + _make_span(result[idx:idx + len(original)], cv["verdict"])
                            + result[idx + len(original):]
                        )
                parts.append(result)
            continue

        # ── Fallback: word similarity (when "original" field is absent) ───
        sentence_matches = {
            cv["verdict"]
            for cv in claim_verdicts
            if cv["verdict"] in _VERDICT_STYLE and _word_similarity(sentence, cv["claim"]) >= 0.18
        }

        if len(sentence_matches) == 1:
            parts.append(_make_span(sentence, sentence_matches.pop()))
            continue

        fragments = _split_at_conjunction(sentence)
        if len(fragments) > 1:
            highlighted = []
            any_matched = False
            for fragment in fragments:
                verdict = _best_verdict(fragment, claim_verdicts, threshold=0.10)
                if verdict:
                    highlighted.append(_make_span(fragment, verdict))
                    any_matched = True
                else:
                    highlighted.append(html.escape(fragment))
            if any_matched:
                parts.append(" ".join(highlighted))
                continue

        if sentence_matches:
            verdict = _best_verdict(sentence, claim_verdicts, threshold=0.18)
            parts.append(_make_span(sentence, verdict) if verdict else html.escape(sentence))
        else:
            parts.append(html.escape(sentence))

    return " ".join(parts)


def md_to_html(text: str) -> str:
    text = html.escape(text)
    # Convert ### / ## / # headings to bold
    text = re.sub(r'^#{1,3}\s+(.+)$', r'<strong>\1</strong>', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    text = re.sub(r'`(.+?)`', r'<code>\1</code>', text)
    text = text.replace('\n\n', '<br><br>').replace('\n', '<br>')
    return text


# ── Inputs ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">🖼️ Chart Image</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "chart_upload",
    type=["png", "jpg", "jpeg", "webp"],
    label_visibility="collapsed",
)

st.markdown('<div class="section-label" style="margin-top:1rem;">✏️ Student Narrative</div>', unsafe_allow_html=True)
narrative = st.text_area(
    "narrative_input",
    height=140,
    placeholder="Enter the student's written interpretation of the chart...",
    label_visibility="collapsed",
)

run_btn = st.button(
    "🔍 Run Fact-Check",
    type="primary",
    disabled=not (uploaded_file and narrative.strip()),
)

# ── Pipeline ──────────────────────────────────────────────────────────────────
if run_btn:
    suffix = os.path.splitext(uploaded_file.name)[1] or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.image(uploaded_file, caption="📊 Uploaded Chart", use_container_width=True)

        with st.status("⚙️ Running pipeline...", expanded=True) as status:
            st.markdown('<p class="step">🧩 Agent 1 — Decomposing narrative into atomic claims and classifying relevance</p>', unsafe_allow_html=True)
            st.markdown('<p class="step">🔎 Agents 2 & 3 — Gathering visual and structured evidence in parallel</p>', unsafe_allow_html=True)
            st.markdown('<p class="step">⚖️ Agent 4 — Resolving verdicts and generating feedback</p>', unsafe_allow_html=True)

            from chart_verifier.orchestrator import run_pipeline

            feedback, claim_verdicts = asyncio.run(run_pipeline(narrative.strip(), tmp_path))
            feedback = re.sub(r'\n?---+\n?', '\n', feedback).strip()
            status.update(label="✅ Analysis complete", state="complete")

        # ── Highlighted Narrative ─────────────────────────────────────────────
        st.markdown('<div class="section-label" style="margin-top:1.5rem;">📝 Narrative — Sentence Verdicts</div>', unsafe_allow_html=True)
        st.markdown("""
<div class="verdict-legend">
  <span class="legend-item"><span class="legend-dot" style="background:#22c55e"></span>Supported</span>
  <span class="legend-item"><span class="legend-dot" style="background:#ef4444"></span>Contradicted</span>
  <span class="legend-item"><span class="legend-dot" style="background:#eab308"></span>Unrelated</span>
</div>
""", unsafe_allow_html=True)
        highlighted = highlight_narrative(narrative.strip(), claim_verdicts)
        st.markdown(
            f'<div class="narrative-highlight-box">{highlighted}</div>',
            unsafe_allow_html=True,
        )

        # ── Feedback ──────────────────────────────────────────────────────────
        st.markdown('<div class="section-label" style="margin-top:1.5rem;">💬 Feedback</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="feedback-box">{md_to_html(feedback)}</div>',
            unsafe_allow_html=True,
        )

    except Exception as exc:
        st.error(f"❌ Pipeline failed: {exc}")
        raise
    finally:
        os.unlink(tmp_path)
