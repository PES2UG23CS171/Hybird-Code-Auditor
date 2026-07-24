# Hybrid Code Auditor

A hybrid static-analysis + LLM refactoring system for Python. It combines deterministic code analysis (AST, cyclomatic complexity, pylint) with retrieval-grounded LLM refactoring in a generate → critique → verify agent loop, behind a Streamlit UI.

**Live demo:** [hybird-code-auditor.streamlit.app](https://hybird-code-auditor-bnszutjrt5hfwsqvhwifbt.streamlit.app)

## How it works

```
Python source
     │
     ▼
1. Static analysis ──── AST walk + radon cyclomatic complexity + pylint issues
     │                  → flags functions above the complexity threshold
     ▼
2. Retrieval (RAG) ──── TF-IDF vectors over PEP 8 / OWASP guidance chunks,
     │                  searched with a FAISS inner-product index
     ▼
3. Agent loop ────────  Generator LLM proposes a refactor
     │                  Critic LLM reviews and improves it
     ▼
4. Verifier (deterministic) ── rejects any candidate that:
     │                  • is not valid Python syntax
     │                  • still exceeds the complexity threshold
     │                  • changes LOC by more than 20%
     │                  Rejected candidates are never adopted; the verifier's
     │                  reason is fed back to the generator for the next round.
     ▼
Refactored source + diff + metrics
```

Key design points:

- **Deterministic guardrails around a nondeterministic model.** The LLM only ever proposes; acceptance is decided by AST parsing and measured complexity, so the output can never be syntactically broken.
- **Local-first.** Runs fully offline against a local Ollama model. Also supports any OpenAI-compatible hosted API (Groq, OpenRouter, Together, vLLM) for cloud deployment.
- **Graceful degradation.** With no LLM reachable, the app still delivers the static-analysis report and retrieved guidelines in analysis-only mode.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The LLM backend is picked automatically:

- **Hosted API:** if an `LLM_API_KEY` secret/environment variable is set (with optional `LLM_BASE_URL` and `LLM_MODEL`), any OpenAI-compatible endpoint is used.
- **Ollama (fully local):** otherwise, a local [Ollama](https://ollama.com) server is used when reachable, with whatever model is installed (`ollama pull llama3.1`).
- **Analysis-only:** with neither available, the app still reports complexity, pylint findings, and retrieved guidelines.

## Deploy (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), create an app pointing at `app.py`.
3. (Optional, enables full refactoring in the cloud) In the app's **Settings → Secrets**, add:

```toml
LLM_BASE_URL = "https://api.groq.com/openai/v1"
LLM_MODEL = "llama-3.1-8b-instant"
LLM_API_KEY = "your-key"
```

Without secrets the deployed app runs in analysis-only mode (complexity report, pylint findings, and retrieved guidelines).

## Project layout

```
app.py                        Streamlit UI
hybrid_code_auditor/
  analysis.py                 AST walk, radon complexity, pylint integration
  rag.py                      Guideline chunking, TF-IDF + FAISS retrieval
  llm.py                      Ollama and OpenAI-compatible clients
  prompts.py                  Generator and critic prompts
  pipeline.py                 Generate → critique → verify loop
  models.py                   Dataclasses for findings and results
data/guidelines/              Local PEP 8 / OWASP guidance corpus
```

## Notes

- The bundled guidance files are concise local summaries intended for retrieval grounding.
- The verifier checks syntactic validity and measured complexity; deeper semantic validation should be backed by project tests.
