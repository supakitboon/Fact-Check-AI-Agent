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

/* Page background */
.stApp { background-color: #f8f9fb; }

/* Main container width */
.block-container { max-width: 780px; padding-top: 2.5rem; padding-bottom: 3rem; }

/* Page header */
.page-header {
    border-bottom: 2px solid #1e3a5f;
    padding-bottom: 0.75rem;
    margin-bottom: 1.75rem;
}
.page-header h1 {
    font-size: 1.75rem;
    font-weight: 700;
    color: #1e3a5f;
    margin: 0 0 0.25rem 0;
    letter-spacing: -0.3px;
}
.page-header p {
    font-size: 0.9rem;
    color: #64748b;
    margin: 0;
}

/* Section card */
.card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.25rem;
}

/* Section label */
.section-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 0.5rem;
}

/* Legend badges */
.legend {
    display: flex;
    gap: 8px;
    margin-bottom: 1rem;
    flex-wrap: wrap;
}
.badge {
    font-size: 0.78rem;
    font-weight: 600;
    padding: 3px 12px;
    border-radius: 20px;
    letter-spacing: 0.02em;
}

/* Narrative highlight container */
.narrative-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    line-height: 2;
    font-size: 1rem;
    margin-bottom: 1.25rem;
}

/* Feedback box */
.feedback-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #1e3a5f;
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    font-size: 0.95rem;
    color: #1a1a1a;
    line-height: 1.75;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 1.5rem 0;
}

/* Step list in status */
.step { font-size: 0.9rem; color: #374151; padding: 2px 0; }

/* Primary button override */
.stButton > button {
    background-color: #1e3a5f !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    padding: 0.45rem 1.5rem !important;
    font-size: 0.9rem !important;
}
.stButton > button:hover {
    background-color: #16304f !important;
}
.stButton > button:disabled {
    background-color: #94a3b8 !important;
    cursor: not-allowed !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 1.5px dashed #cbd5e1 !important;
    border-radius: 8px !important;
    padding: 0.5rem !important;
}

/* Text area */
textarea {
    border-radius: 6px !important;
    border-color: #cbd5e1 !important;
    font-size: 0.95rem !important;
}

/* Status box */
[data-testid="stStatus"] {
    border-radius: 8px !important;
    border-color: #e2e8f0 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1>Chart Fact-Checker</h1>
    <p>Upload a chart and enter a student narrative. The pipeline will verify each claim against the chart data.</p>
</div>
""", unsafe_allow_html=True)

# ── Highlighting helpers ──────────────────────────────────────────────────────

VERDICT_STYLE = {
    "supported":    {"bg": "#dcfce7", "color": "#14532d", "label": "Correct"},
    "contradicted": {"bg": "#fee2e2", "color": "#7f1d1d", "label": "Incorrect"},
    "unrelated":    {"bg": "#ede9fe", "color": "#4c1d95", "label": "Unrelated"},
}

VERDICT_PRIORITY = {"contradicted": 3, "unrelated": 2, "supported": 1, "unknown": 0}

_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "have",
    "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "can", "in", "on", "at", "to", "for", "of", "and",
    "or", "but", "not", "with", "from", "by", "as", "this", "that",
    "it", "its", "they", "their", "which", "who", "over", "about",
    "into", "than", "so", "also", "just", "only",
}


def _key_words(text: str) -> set[str]:
    words = re.sub(r"[^\w\s]", "", text).lower().split()
    return {w for w in words if w not in _STOP_WORDS and len(w) > 1}


def _overlap(sentence: str, claim: str) -> float:
    """Fraction of claim's key words that appear in the sentence."""
    s_words = _key_words(sentence)
    c_words = _key_words(claim)
    if not c_words:
        return 0.0
    return len(s_words & c_words) / len(c_words)


def _sentence_verdict(sentence: str, claim_verdicts: list[dict]) -> str:
    best, best_score = "unknown", 0
    for cv in claim_verdicts:
        score = _overlap(sentence, cv["claim"])
        if score >= 0.35:
            v = cv["verdict"]
            if VERDICT_PRIORITY.get(v, 0) > best_score:
                best, best_score = v, VERDICT_PRIORITY[v]
    return best


def highlight_narrative(narrative: str, claim_verdicts: list[dict]) -> str:
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", narrative.strip())
    parts = []
    for sentence in sentences:
        if not sentence.strip():
            continue
        verdict = _sentence_verdict(sentence, claim_verdicts)
        style = VERDICT_STYLE.get(verdict)
        if style:
            parts.append(
                f'<span style="background-color:{style["bg"]};color:{style["color"]};'
                f'padding:2px 6px;border-radius:4px;margin:1px;">'
                f'{sentence}</span>'
            )
        else:
            parts.append(f'<span style="color:#1a1a1a;">{sentence}</span>')
    return " ".join(parts)


def md_to_html(text: str) -> str:
    """Convert basic markdown to HTML for rendering inside raw HTML divs."""
    text = html.escape(text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    text = re.sub(r'`(.+?)`', r'<code>\1</code>', text)
    text = text.replace('\n\n', '<br><br>').replace('\n', '<br>')
    return text


def legend_html() -> str:
    items = [
        ("Correct",   "#dcfce7", "#14532d"),
        ("Incorrect", "#fee2e2", "#7f1d1d"),
        ("Unrelated", "#ede9fe", "#4c1d95"),
    ]
    badges = "".join(
        f'<span class="badge" style="background:{bg};color:{tc};">{label}</span>'
        for label, bg, tc in items
    )
    return f'<div class="legend">{badges}</div>'


# ── Inputs ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Chart Image</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "chart_upload",
    type=["png", "jpg", "jpeg", "webp"],
    label_visibility="collapsed",
)

st.markdown('<div class="section-label" style="margin-top:1rem;">Student Narrative</div>', unsafe_allow_html=True)
narrative = st.text_area(
    "narrative_input",
    height=140,
    placeholder="Enter the student's written interpretation of the chart...",
    label_visibility="collapsed",
)

run_btn = st.button(
    "Run Fact-Check",
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
        st.image(uploaded_file, caption="Uploaded Chart", use_container_width=True)

        with st.status("Running pipeline...", expanded=True) as status:
            st.markdown('<p class="step">Agent 1 — Decomposing narrative into atomic claims and classifying relevance</p>', unsafe_allow_html=True)
            st.markdown('<p class="step">Agents 2 & 3 — Gathering visual and structured evidence in parallel</p>', unsafe_allow_html=True)
            st.markdown('<p class="step">Agent 4 — Resolving verdicts and generating feedback</p>', unsafe_allow_html=True)

            from chart_verifier.orchestrator import run_pipeline

            feedback, claim_verdicts = asyncio.run(run_pipeline(narrative.strip(), tmp_path))
            feedback = re.sub(r'\n?---+\n?', '\n', feedback).strip()
            status.update(label="Analysis complete", state="complete")

        # ── Narrative analysis ────────────────────────────────────────────────
        st.markdown('<div class="section-label" style="margin-top:1.5rem;">Narrative Analysis</div>', unsafe_allow_html=True)
        st.markdown(legend_html(), unsafe_allow_html=True)
        highlighted = highlight_narrative(narrative.strip(), claim_verdicts)
        st.markdown(
            f'<div class="narrative-box">{highlighted}</div>',
            unsafe_allow_html=True,
        )

        # ── Feedback ──────────────────────────────────────────────────────────
        st.markdown('<div class="section-label">Feedback</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="feedback-box">{md_to_html(feedback)}</div>',
            unsafe_allow_html=True,
        )

    except Exception as exc:
        st.error(f"Pipeline failed: {exc}")
        raise
    finally:
        os.unlink(tmp_path)
