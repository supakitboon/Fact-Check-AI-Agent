"""
Streamlit UI for the Chart Verifier agent pipeline.

Run with:
    streamlit run streamlit_app.py
"""

import asyncio
import tempfile
import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chart Fact-Checker",
    page_icon="📊",
    layout="centered",
)

st.title("📊 Chart Fact-Checker")
st.caption(
    "Upload a chart image and enter a student narrative — the 4-agent pipeline "
    "will fact-check every claim and return structured feedback."
)

# ── Inputs ────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Chart image (PNG / JPG / WebP)",
    type=["png", "jpg", "jpeg", "webp"],
)

narrative = st.text_area(
    "Student narrative",
    height=160,
    placeholder=(
        "e.g. Revenue grew steadily from 2020 to 2022, "
        "and Product A had the highest sales throughout the period."
    ),
)

run_btn = st.button("Run fact-check", type="primary", disabled=not (uploaded_file and narrative.strip()))

# ── Pipeline ──────────────────────────────────────────────────────────────────
if run_btn:
    suffix = os.path.splitext(uploaded_file.name)[1] or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        st.image(uploaded_file, caption="Uploaded chart", use_container_width=True)

        with st.status("Running 4-agent pipeline…", expanded=True) as status:
            st.write("📋 Agent 1 — Decomposing narrative into atomic claims & classifying relevance…")
            st.write("🔍 Agents 2 & 3 — Gathering visual evidence and structured data in parallel…")
            st.write("⚖️ Agent 4 — Resolving verdicts and generating student feedback…")

            from chart_verifier.orchestrator import run_pipeline

            feedback = asyncio.run(run_pipeline(narrative.strip(), tmp_path))
            status.update(label="Done!", state="complete")

        st.subheader("Student Feedback")
        st.markdown(feedback)

    except Exception as exc:
        st.error(f"Pipeline failed: {exc}")
        raise
    finally:
        os.unlink(tmp_path)
