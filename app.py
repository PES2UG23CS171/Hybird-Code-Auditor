from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

from hybrid_code_auditor.config import AuditorConfig
from hybrid_code_auditor.llm import OllamaClient, OpenAICompatibleClient
from hybrid_code_auditor.pipeline import HybridCodeAuditor
from hybrid_code_auditor.rag import build_default_guideline_store


st.set_page_config(page_title="Hybrid Code Auditor", layout="wide")


def _secret(name: str, default: str = "") -> str:
    if name in os.environ:
        return os.environ[name]
    try:
        return str(st.secrets.get(name, default))
    except (FileNotFoundError, AttributeError):
        return default


def _installed_ollama_models(host: str) -> list[str]:
    try:
        import requests

        response = requests.get(f"{host.rstrip('/')}/api/tags", timeout=3)
        response.raise_for_status()
        return [model["name"] for model in response.json().get("models", [])]
    except Exception:
        return []


def _make_client():
    """Pick the LLM backend automatically.

    A configured LLM_API_KEY secret selects the hosted OpenAI-compatible
    backend; otherwise a local Ollama server is used when reachable. The
    key never leaves the server.
    """
    api_key = _secret("LLM_API_KEY")
    if api_key:
        return OpenAICompatibleClient(
            base_url=_secret("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
            api_key=api_key,
            model_name=_secret("LLM_MODEL", "llama-3.1-8b-instant"),
        )
    host = _secret("OLLAMA_HOST", "http://localhost:11434")
    models = _installed_ollama_models(host)
    model_name = _secret("OLLAMA_MODEL") or (models[0] if models else "llama3.2")
    return OllamaClient(base_url=host, model_name=model_name)


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


@st.cache_resource
def _guideline_store():
    return build_default_guideline_store()


st.title("Hybrid Code Auditor")
st.caption(
    "Local-first refactoring pipeline: AST + pylint/radon analysis, FAISS guideline retrieval, "
    "and an LLM generate-critique-verify loop."
)

client = _make_client()

with st.sidebar:
    st.header("Configuration")
    complexity_threshold = st.number_input("Complexity threshold", min_value=1, max_value=50, value=10)
    max_rounds = st.number_input("Max rounds", min_value=1, max_value=10, value=5)
    if client.is_available():
        backend_label = "hosted API" if isinstance(client, OpenAICompatibleClient) else "local Ollama"
        st.success(f"Model: {client.model_name} ({backend_label})")
    else:
        st.warning("No LLM reachable — running in analysis-only mode.")
    st.divider()
    st.write(
        "The bundled local guidance is built from concise summaries of PEP 8 "
        "and OWASP-oriented secure coding notes."
    )

uploaded_file = st.file_uploader("Upload a Python file", type=["py"])
if uploaded_file is not None:
    source_code = uploaded_file.getvalue().decode("utf-8", errors="replace")
    st.code(source_code, language="python")
else:
    source_code = st.text_area("Source code", value=_default_sample(), height=300)

auditor = HybridCodeAuditor(
    config=AuditorConfig(
        complexity_threshold=int(complexity_threshold),
        max_rounds=int(max_rounds),
    ),
    guideline_store=_guideline_store(),
    client=client,
)

run_button = st.button("Analyze and Refactor", type="primary")

if run_button:
    if not source_code.strip():
        st.error("Please provide some Python code to analyze.")
        st.stop()

    with st.spinner("Analyzing and refactoring..."):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "input.py"
            temp_path.write_text(source_code, encoding="utf-8")
            result = auditor.refactor_file(temp_path)

    st.subheader("Verification")
    st.info(result.summary.verification_message)

    metrics = st.columns(4)
    metrics[0].metric("Candidates", len(result.function_reports))
    metrics[1].metric("CC before", result.summary.complexity_before)
    metrics[2].metric("CC after", result.summary.complexity_after)
    metrics[3].metric("Rounds used", result.summary.rounds_used)

    if result.function_reports:
        st.subheader("Flagged Functions")
        st.dataframe(
            [
                {
                    "Function": finding.name,
                    "Lines": f"{finding.lineno}-{finding.end_lineno}",
                    "Cyclomatic complexity": finding.complexity,
                    "Pylint issues": len(finding.pylint_issues),
                }
                for finding in result.function_reports
            ],
            use_container_width=True,
        )
        for finding in result.function_reports:
            if finding.pylint_issues:
                with st.expander(f"Pylint issues in {finding.name}"):
                    for issue in finding.pylint_issues:
                        symbol = f" ({issue.symbol})" if issue.symbol else ""
                        st.write(f"Line {issue.line}: {issue.message}{symbol}")

    left, right = st.columns(2)
    with left:
        st.subheader("Original")
        st.code(result.original_code, language="python")
    with right:
        st.subheader("Refactored")
        st.code(result.refactored_code, language="python")

    st.subheader("Diff")
    st.code(result.diff, language="diff")

    if result.summary.guideline_hits:
        st.subheader("Retrieved Guidelines")
        for item in result.summary.guideline_hits:
            with st.expander(f"{item.function_name} - top guidelines"):
                for chunk in item.chunks:
                    st.markdown(f"**{chunk.source}** — score {chunk.score:.3f}")
                    st.write(chunk.text)
