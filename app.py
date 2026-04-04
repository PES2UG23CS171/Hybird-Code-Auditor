from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from hybrid_code_auditor.config import AuditorConfig
from hybrid_code_auditor.pipeline import HybridCodeAuditor
from hybrid_code_auditor.rag import build_default_guideline_store


st.set_page_config(page_title="Hybrid Code Auditor", layout="wide")


def _default_sample() -> str:
    return """def process_items(items):
    total = 0
    for item in items:
        if item is None:
            continue
        if isinstance(item, dict):
            if item.get('enabled') and item.get('value') is not None:
                if item.get('value') > 0:
                    total += item['value']
        elif isinstance(item, int):
            if item > 0:
                total += item
    return total
"""


st.title("Hybrid Code Auditor")
st.caption("Fully local refactoring pipeline with AST analysis, FAISS retrieval, and Ollama-driven generation.")

with st.sidebar:
    st.header("Configuration")
    model_name = st.text_input("Ollama model", value="llama3.1:latest")
    ollama_host = st.text_input("Ollama host", value="http://localhost:11434")
    complexity_threshold = st.number_input("Complexity threshold", min_value=1, max_value=50, value=10)
    max_rounds = st.number_input("Max rounds", min_value=1, max_value=10, value=5)
    st.divider()
    st.write("The bundled local guidance is built from concise summaries of PEP 8 and OWASP-oriented secure coding notes.")

uploaded_file = st.file_uploader("Upload a Python file", type=["py"])
source_code = st.text_area("Source code", value=_default_sample(), height=300)

if uploaded_file is not None:
    source_code = uploaded_file.read().decode("utf-8")

if "auditor" not in st.session_state:
    guideline_store = build_default_guideline_store()
    st.session_state.auditor = HybridCodeAuditor(
        config=AuditorConfig(
            model_name=model_name,
            ollama_host=ollama_host,
            complexity_threshold=int(complexity_threshold),
            max_rounds=int(max_rounds),
        ),
        guideline_store=guideline_store,
    )

st.session_state.auditor.config.model_name = model_name
st.session_state.auditor.config.ollama_host = ollama_host
st.session_state.auditor.config.complexity_threshold = int(complexity_threshold)
st.session_state.auditor.config.max_rounds = int(max_rounds)
st.session_state.auditor.client.model_name = model_name
st.session_state.auditor.client.base_url = ollama_host

run_button = st.button("Analyze and Refactor", type="primary")

if run_button:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "input.py"
        temp_path.write_text(source_code, encoding="utf-8")

        result = st.session_state.auditor.refactor_file(temp_path)

    left, right = st.columns(2)
    with left:
        st.subheader("Original")
        st.code(result.original_code, language="python")
    with right:
        st.subheader("Refactored")
        st.code(result.refactored_code, language="python")

    metrics = st.columns(4)
    metrics[0].metric("Candidates", len(result.function_reports))
    metrics[1].metric("CC before", result.summary.complexity_before)
    metrics[2].metric("CC after", result.summary.complexity_after)
    metrics[3].metric("Rounds used", result.summary.rounds_used)

    st.subheader("Diff")
    st.code(result.diff, language="diff")

    st.subheader("Retrieved Guidelines")
    for item in result.summary.guideline_hits:
        with st.expander(f"{item.function_name} - top guidelines"):
            for chunk in item.chunks:
                st.markdown(f"**Score:** {chunk.score:.3f}")
                st.write(chunk.text)

    st.subheader("Verification")
    st.write(result.summary.verification_message)
