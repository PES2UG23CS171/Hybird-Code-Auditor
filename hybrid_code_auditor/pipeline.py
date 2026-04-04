from __future__ import annotations

import ast
import difflib
from pathlib import Path

from .analysis import analyze_python_source, find_refactor_candidates
from .config import AuditorConfig
from .llm import OllamaClient
from .models import (
    GuidelineRetrieval,
    RefactorResult,
    RefactorSummary,
    VerificationResult,
)
from .prompts import build_critic_prompt, build_generator_prompt
from .rag import GuidelineStore


class HybridCodeAuditor:
    def __init__(self, config: AuditorConfig, guideline_store: GuidelineStore) -> None:
        self.config = config
        self.guideline_store = guideline_store
        self.client = OllamaClient(base_url=config.ollama_host, model_name=config.model_name)

    def refactor_file(self, source_path: Path) -> RefactorResult:
        original_code = source_path.read_text(encoding="utf-8")
        candidates = find_refactor_candidates(original_code, self.config.complexity_threshold, source_path)

        if not candidates:
            summary = RefactorSummary(
                complexity_before=0,
                complexity_after=0,
                rounds_used=0,
                guideline_hits=[],
                verification_message="No functions exceeded the current thresholds.",
            )
            return RefactorResult(
                original_code=original_code,
                refactored_code=original_code,
                diff="No changes.",
                summary=summary,
                function_reports=[],
                source_path=source_path,
            )

        function_reports = candidates
        retrieved_guidelines: list[GuidelineRetrieval] = []
        working_code = original_code
        rounds_used = 0
        total_before = max((function.complexity for function in analyze_python_source(original_code)), default=0)
        total_after = total_before
        verification_message = ""

        for round_index in range(self.config.max_rounds):
            rounds_used = round_index + 1
            current_candidates = find_refactor_candidates(working_code, self.config.complexity_threshold)
            if not current_candidates:
                verification_message = "No functions exceeded the current thresholds after refactoring."
                break

            target_function = max(current_candidates, key=lambda item: item.complexity)
            top_guidelines = self.guideline_store.search(
                query=f"function {target_function.name} complexity refactor security",
                top_k=self.config.top_k_guidelines,
            )
            retrieved_guidelines.append(
                GuidelineRetrieval(function_name=target_function.name, chunks=top_guidelines)
            )

            function_source = _extract_function_source(working_code, target_function.lineno, target_function.end_lineno)
            generator_prompt = build_generator_prompt(
                function=target_function,
                source_code=function_source,
                guidelines=top_guidelines,
            )
            generated_code = self._generate_or_fallback(generator_prompt, working_code)

            critic_prompt = build_critic_prompt(
                function_name=target_function.name,
                candidate_code=generated_code,
                guidelines=top_guidelines,
            )
            critiqued_code = self._generate_or_fallback(critic_prompt, generated_code)
            candidate_code = _sanitize_model_output(critiqued_code or generated_code or working_code)

            candidate_full_code = _replace_function_block(working_code, target_function.lineno, target_function.end_lineno, candidate_code)
            verification = self._verify_candidate(working_code, candidate_full_code)
            total_after = verification.complexity_after
            verification_message = verification.message
            if verification.passed:
                working_code = candidate_full_code
                break

            working_code = candidate_full_code

        diff = "\n".join(
            difflib.unified_diff(
                original_code.splitlines(),
                working_code.splitlines(),
                fromfile="original.py",
                tofile="refactored.py",
                lineterm="",
            )
        )

        summary = RefactorSummary(
            complexity_before=total_before,
            complexity_after=total_after,
            rounds_used=rounds_used,
            guideline_hits=retrieved_guidelines,
            verification_message=verification_message,
        )
        return RefactorResult(
            original_code=original_code,
            refactored_code=working_code,
            diff=diff,
            summary=summary,
            function_reports=function_reports,
            source_path=source_path,
        )

    def _verify_candidate(self, original_code: str, candidate_code: str) -> VerificationResult:
        original_lines = [line for line in original_code.splitlines() if line.strip()]
        candidate_lines = [line for line in candidate_code.splitlines() if line.strip()]
        loc_before = len(original_lines)
        loc_after = len(candidate_lines)
        loc_change_ratio = abs(loc_after - loc_before) / max(loc_before, 1)

        try:
            ast.parse(candidate_code)
        except SyntaxError as exc:
            return VerificationResult(
                passed=False,
                complexity_before=0,
                complexity_after=999,
                loc_before=loc_before,
                loc_after=loc_after,
                message=f"Rejected: candidate is not valid Python syntax ({exc.msg}).",
            )

        candidate_functions = analyze_python_source(candidate_code)
        complexity_after = max((item.complexity for item in candidate_functions), default=0)
        passed = complexity_after <= self.config.complexity_threshold and loc_change_ratio <= self.config.max_loc_change_ratio
        if passed:
            message = f"Accepted: complexity {complexity_after} and LOC change {loc_change_ratio:.0%} are within limits."
        else:
            message = f"Rejected: complexity {complexity_after} or LOC change {loc_change_ratio:.0%} exceeded limits."

        return VerificationResult(
            passed=passed,
            complexity_before=0,
            complexity_after=complexity_after,
            loc_before=loc_before,
            loc_after=loc_after,
            message=message,
        )

    def _generate_or_fallback(self, prompt: str, fallback: str) -> str:
        try:
            return self.client.generate(prompt)
        except Exception as exc:
            return fallback


def _replace_function_block(source_code: str, start_line: int, end_line: int, replacement_block: str) -> str:
    lines = source_code.splitlines()
    before = lines[: max(0, start_line - 1)]
    after = lines[min(len(lines), end_line) :]
    replacement_lines = replacement_block.splitlines()
    combined = before + replacement_lines + after
    return "\n".join(combined) + ("\n" if source_code.endswith("\n") or replacement_block.endswith("\n") else "")


def _sanitize_model_output(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```"):
            if lines[-1].startswith("```"):
                return "\n".join(lines[1:-1]).strip()
        return "\n".join(line for line in lines if not line.startswith("```" )).strip()
    return stripped


def _extract_function_source(source_code: str, start_line: int, end_line: int) -> str:
    lines = source_code.splitlines()
    start_index = max(0, start_line - 1)
    end_index = min(len(lines), end_line)
    return "\n".join(lines[start_index:end_index])
