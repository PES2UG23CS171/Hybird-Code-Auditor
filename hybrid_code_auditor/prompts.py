from __future__ import annotations

from typing import Iterable

from .models import FunctionFinding, GuidelineChunk


def build_generator_prompt(function: FunctionFinding, source_code: str, guidelines: Iterable[GuidelineChunk], feedback: str = "") -> str:
    guideline_text = "\n\n".join(
        f"[Source: {chunk.source}]\n{chunk.text}"
        for chunk in guidelines
    )
    return f"""You are a senior Python refactoring assistant.

Goal:
- Refactor the target function to reduce cyclomatic complexity below 10.
- Preserve behavior and inputs/outputs.
- Follow PEP 8 and secure coding guidance.
- Keep the code local and explainable.

Target function:
Name: {function.name}
Location: lines {function.lineno}-{function.end_lineno}
Current cyclomatic complexity: {function.complexity}

Function source:
{source_code}

Retrieved guidance:
{guideline_text}

Feedback from prior round:
{feedback or 'None'}

Return only the replacement Python code block for this function, plus any helper functions it needs. Do not add prose.
"""


def build_critic_prompt(function_name: str, candidate_code: str, guidelines: Iterable[GuidelineChunk]) -> str:
    guideline_text = "\n\n".join(
        f"[Source: {chunk.source}]\n{chunk.text}"
        for chunk in guidelines
    )
    return f"""You are a code critic.

Review the candidate refactor for function {function_name}.
Check for:
- unnecessary branching
- readability problems
- missed security hygiene from the retrieved guidance
- any sign that behavior was made ambiguous

Candidate code:
{candidate_code}

Guidance:
{guideline_text}

Return an improved replacement code block only.
"""
