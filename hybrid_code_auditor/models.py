from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Issue:
    line: int
    message: str
    symbol: str | None = None


@dataclass
class FunctionFinding:
    name: str
    lineno: int
    end_lineno: int
    complexity: int
    pylint_issues: list[Issue] = field(default_factory=list)

    @property
    def has_quality_issue(self) -> bool:
        return bool(self.pylint_issues)


@dataclass
class GuidelineChunk:
    text: str
    source: str
    score: float = 0.0


@dataclass
class GuidelineRetrieval:
    function_name: str
    chunks: list[GuidelineChunk]


@dataclass
class VerificationResult:
    passed: bool
    complexity_before: int
    complexity_after: int
    loc_before: int
    loc_after: int
    message: str


@dataclass
class RefactorSummary:
    complexity_before: int
    complexity_after: int
    rounds_used: int
    guideline_hits: list[GuidelineRetrieval]
    verification_message: str


@dataclass
class RefactorResult:
    original_code: str
    refactored_code: str
    diff: str
    summary: RefactorSummary
    function_reports: list[FunctionFinding]
    source_path: Path
