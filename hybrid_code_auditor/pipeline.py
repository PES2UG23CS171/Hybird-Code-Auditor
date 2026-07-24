from __future__ import annotations

import ast
import difflib
import re
from pathlib import Path

from .analysis import analyze_python_source, find_refactor_candidates
from .config import AuditorConfig
from .llm import LLMClient, OllamaClient
from .models import (
    GuidelineRetrieval,
    RefactorResult,
    RefactorSummary,
    VerificationResult,
)
from .prompts import build_critic_prompt, build_generator_prompt
from .rag import GuidelineStore


class HybridCodeAuditor:
    def __init__(
        self,
        config: AuditorConfig,
        guideline_store: GuidelineStore,
        client: LLMClient | None = None,
    ) -> None:
        self.config = config
        self.guideline_store = guideline_store
        self.client: LLMClient = client or OllamaClient(
            base_url=config.ollama_host, model_name=config.model_name
        )

    def refactor_file(self, source_path: Path) -> RefactorResult:
        original_code = source_path.read_text(encoding="utf-8")

        try:
            all_findings = analyze_python_source(original_code)
        except SyntaxError as exc:
            return self._result(
                original_code,
                original_code,
                source_path,
                function_reports=[],
                guideline_hits=[],
                rounds_used=0,
                complexity_before=0,
                complexity_after=0,
                message=f"Input is not valid Python: line {exc.lineno}: {exc.msg}",
            )

        candidates = [
            finding
            for finding in all_findings
            if finding.complexity > self.config.complexity_threshold or finding.has_quality_issue
        ]
        total_before = max((finding.complexity for finding in all_findings), default=0)

        if not candidates:
            return self._result(
                original_code,
                original_code,
                source_path,
                function_reports=[],
                guideline_hits=[],
                rounds_used=0,
                complexity_before=total_before,
                complexity_after=total_before,
                message="No functions exceeded the current thresholds.",
            )

        if not self.client.is_available():
            return self._result(
                original_code,
                original_code,
                source_path,
                function_reports=candidates,
                guideline_hits=self._retrieve_for(candidates),
                rounds_used=0,
                complexity_before=total_before,
                complexity_after=total_before,
                message=(
                    "Analysis-only mode: no LLM backend is reachable, so candidates and "
                    "retrieved guidelines are shown without automated refactoring. "
                    "Connect Ollama or a hosted API to enable refactoring."
                ),
            )

        retrieved_guidelines: list[GuidelineRetrieval] = []
        working_code = original_code
        rounds_used = 0
        verification_message = ""
        feedback = ""

        for round_index in range(self.config.max_rounds):
            rounds_used = round_index + 1
            current_candidates = find_refactor_candidates(working_code, self.config.complexity_threshold)
            if not current_candidates:
                verification_message = "All functions are within thresholds after refactoring."
                break

            target_function = max(current_candidates, key=lambda item: item.complexity)
            top_guidelines = self.guideline_store.search(
                query=f"function {target_function.name} complexity refactor security",
                top_k=self.config.top_k_guidelines,
            )
            retrieved_guidelines.append(
                GuidelineRetrieval(function_name=target_function.name, chunks=top_guidelines)
            )

            function_source = _extract_function_source(
                working_code, target_function.lineno, target_function.end_lineno
            )
            generator_prompt = build_generator_prompt(
                function=target_function,
                source_code=function_source,
                guidelines=top_guidelines,
                feedback=feedback,
            )
            try:
                generated_code = _sanitize_model_output(self.client.generate(generator_prompt))
                if not _parses(generated_code):
                    verification_message = "Rejected: model output was not valid Python syntax."
                    feedback = (
                        "Rejected: the previous response was not valid Python. Respond with exactly "
                        "one fenced Python code block containing only the replacement code."
                    )
                    continue
                critic_prompt = build_critic_prompt(
                    function_name=target_function.name,
                    candidate_code=generated_code,
                    guidelines=top_guidelines,
                )
                critiqued_code = _sanitize_model_output(self.client.generate(critic_prompt))
            except RuntimeError as exc:
                verification_message = f"Stopped: {exc}"
                break

            # Keep the generator's valid output if the critic mangled it.
            candidate_code = critiqued_code if _parses(critiqued_code) else generated_code

            candidate_full_code = _replace_function_block(
                working_code, target_function.lineno, target_function.end_lineno, candidate_code
            )
            verification = self._verify_candidate(working_code, candidate_full_code)
            verification_message = verification.message
            if verification.passed:
                working_code = candidate_full_code
                feedback = ""
            else:
                feedback = verification.message

        try:
            total_after = max(
                (finding.complexity for finding in analyze_python_source(working_code)), default=0
            )
        except SyntaxError:
            # Should not happen since failed candidates are never adopted,
            # but never let the summary computation crash the app.
            total_after = total_before

        return self._result(
            original_code,
            working_code,
            source_path,
            function_reports=candidates,
            guideline_hits=retrieved_guidelines,
            rounds_used=rounds_used,
            complexity_before=total_before,
            complexity_after=total_after,
            message=verification_message,
        )

    def _retrieve_for(self, candidates: list) -> list[GuidelineRetrieval]:
        retrievals: list[GuidelineRetrieval] = []
        for finding in candidates:
            chunks = self.guideline_store.search(
                query=f"function {finding.name} complexity refactor security",
                top_k=self.config.top_k_guidelines,
            )
            retrievals.append(GuidelineRetrieval(function_name=finding.name, chunks=chunks))
        return retrievals

    def _result(
        self,
        original_code: str,
        refactored_code: str,
        source_path: Path,
        *,
        function_reports: list,
        guideline_hits: list[GuidelineRetrieval],
        rounds_used: int,
        complexity_before: int,
        complexity_after: int,
        message: str,
    ) -> RefactorResult:
        if refactored_code == original_code:
            diff = "No changes."
        else:
            diff = "\n".join(
                difflib.unified_diff(
                    original_code.splitlines(),
                    refactored_code.splitlines(),
                    fromfile="original.py",
                    tofile="refactored.py",
                    lineterm="",
                )
            )
        summary = RefactorSummary(
            complexity_before=complexity_before,
            complexity_after=complexity_after,
            rounds_used=rounds_used,
            guideline_hits=guideline_hits,
            verification_message=message,
        )
        return RefactorResult(
            original_code=original_code,
            refactored_code=refactored_code,
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
        complexity_ok = complexity_after <= self.config.complexity_threshold
        loc_change = abs(loc_after - loc_before)
        loc_ok = (
            loc_change_ratio <= self.config.max_loc_change_ratio
            or loc_change <= self.config.loc_change_allowance
        )
        passed = complexity_ok and loc_ok
        if passed:
            message = (
                f"Accepted: complexity {complexity_after} and "
                f"LOC change {loc_change_ratio:.0%} are within limits."
            )
        else:
            reasons = []
            if not complexity_ok:
                reasons.append(
                    f"complexity {complexity_after} exceeds threshold {self.config.complexity_threshold}"
                )
            if not loc_ok:
                reasons.append(
                    f"LOC change {loc_change_ratio:.0%} ({loc_change} lines) exceeds limit "
                    f"{self.config.max_loc_change_ratio:.0%} (allowance {self.config.loc_change_allowance} lines)"
                )
            message = "Rejected: " + " and ".join(reasons) + "."

        return VerificationResult(
            passed=passed,
            complexity_before=0,
            complexity_after=complexity_after,
            loc_before=loc_before,
            loc_after=loc_after,
            message=message,
        )


def _replace_function_block(source_code: str, start_line: int, end_line: int, replacement_block: str) -> str:
    lines = source_code.splitlines()
    before = lines[: max(0, start_line - 1)]
    after = lines[min(len(lines), end_line):]
    replacement_lines = _reindent(replacement_block, _line_indent(lines, start_line - 1))
    combined = before + replacement_lines + after
    return "\n".join(combined) + ("\n" if source_code.endswith("\n") or replacement_block.endswith("\n") else "")


def _line_indent(lines: list[str], index: int) -> str:
    if 0 <= index < len(lines):
        line = lines[index]
        return line[: len(line) - len(line.lstrip())]
    return ""


def _reindent(block: str, target_indent: str) -> list[str]:
    """Shift a replacement block so its outermost lines match the original indent.

    Model output is usually dedented to column zero even when the original
    function was a nested method; without this, splicing breaks the file.
    """
    block_lines = block.splitlines()
    non_empty = [line for line in block_lines if line.strip()]
    if not non_empty:
        return block_lines
    current_indent = min(len(line) - len(line.lstrip()) for line in non_empty)
    result: list[str] = []
    for line in block_lines:
        if not line.strip():
            result.append("")
        else:
            result.append(target_indent + line[current_indent:])
    return result


def _sanitize_model_output(text: str) -> str:
    """Extract Python code from model output that may include prose or fences."""
    stripped = text.strip()

    # Prefer the largest parseable fenced code block anywhere in the output.
    fenced_blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", stripped, flags=re.DOTALL)
    parseable = [block.strip("\n") for block in fenced_blocks if _parses(block)]
    if parseable:
        return max(parseable, key=len)
    if fenced_blocks:
        return max((block.strip("\n") for block in fenced_blocks), key=len)

    if _parses(stripped):
        return stripped

    # Last resort: take everything from the first def/class/import line onward,
    # dropping trailing prose lines until the remainder parses.
    lines = stripped.splitlines()
    for start, line in enumerate(lines):
        if re.match(r"\s*(def |class |import |from |@)", line):
            for end in range(len(lines), start, -1):
                candidate = "\n".join(lines[start:end])
                if _parses(candidate):
                    return candidate
            break
    return stripped


def _parses(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _extract_function_source(source_code: str, start_line: int, end_line: int) -> str:
    lines = source_code.splitlines()
    start_index = max(0, start_line - 1)
    end_index = min(len(lines), end_line)
    return "\n".join(lines[start_index:end_index])
