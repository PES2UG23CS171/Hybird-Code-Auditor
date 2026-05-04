# Hybrid Code Auditor

Local-first code refactoring system for Python that combines AST analysis, static quality checks, retrieval-grounded prompting, and a Streamlit interface.

## What it does

- Finds Python functions with cyclomatic complexity above a threshold or with Pylint issues.
- Builds a local FAISS index over concise PEP 8 and OWASP-inspired guidance chunks.
- Uses a local Ollama model to generate, critique, and verify refactors in a 3-agent loop.
- Enforces verification constraints:
  - cyclomatic complexity below 10
  - LOC change within 20 percent
  - valid Python syntax
- Shows original code, refactored code, diff, complexity reduction, and retrieved guidance in Streamlit.

## Local setup

1. Install Ollama and pull a local model such as `llama3.2`.
2. Start the Ollama server.
3. Install Python dependencies.
4. Run the Streamlit app.

## Run

```bash
pip install -e .
streamlit run app.py
```

## Notes

- The bundled guidance files are concise local summaries intended for retrieval grounding.
- The verifier checks syntactic validity and measured complexity; deeper semantic validation should be backed by project tests.
# Hybird-Code-Auditor
