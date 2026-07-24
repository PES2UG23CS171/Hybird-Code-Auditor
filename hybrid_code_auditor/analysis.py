from __future__ import annotations

import ast
import json
import tempfile
from io import StringIO
from pathlib import Path

from pylint.lint import Run
from pylint.reporters.json_reporter import JSONReporter
from radon.complexity import cc_visit

from .models import FunctionFinding, Issue


class FunctionCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.functions: list[ast.AST] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.functions.append(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.functions.append(node)
        self.generic_visit(node)


def _function_end_lineno(node: ast.AST) -> int:
    end_lineno = getattr(node, "end_lineno", None)
    if end_lineno is not None:
        return int(end_lineno)
    return int(getattr(node, "lineno", 1))


def analyze_python_source(source_code: str, path: Path | None = None) -> list[FunctionFinding]:
    tree = ast.parse(source_code)
    collector = FunctionCollector()
    collector.visit(tree)

    cc_map: dict[tuple[str, int], int] = {}
    for block in cc_visit(source_code):
        cc_map[(block.name, block.lineno)] = block.complexity

    pylint_issues = _collect_pylint_issues(source_code, path)

    findings: list[FunctionFinding] = []
    for node in collector.functions:
        name = getattr(node, "name", "<lambda>")
        def_lineno = int(getattr(node, "lineno", 1))
        decorators = getattr(node, "decorator_list", [])
        start_lineno = min((int(d.lineno) for d in decorators), default=def_lineno)
        end_lineno = _function_end_lineno(node)
        complexity = cc_map.get((name, def_lineno), 1)
        function_issues = [issue for issue in pylint_issues if start_lineno <= issue.line <= end_lineno]
        findings.append(
            FunctionFinding(
                name=name,
                lineno=start_lineno,
                end_lineno=end_lineno,
                complexity=complexity,
                pylint_issues=function_issues,
            )
        )

    return findings


def find_refactor_candidates(source_code: str, threshold: int, path: Path | None = None) -> list[FunctionFinding]:
    findings = analyze_python_source(source_code, path)
    return [finding for finding in findings if finding.complexity > threshold or finding.has_quality_issue]


def _collect_pylint_issues(source_code: str, path: Path | None = None) -> list[Issue]:
    reporter_output = StringIO()
    reporter = JSONReporter(output=reporter_output)

    temp_path: Path | None = None
    try:
        if path is not None and path.exists():
            temp_path = path
        else:
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as temp_file:
                temp_file.write(source_code)
                temp_path = Path(temp_file.name)

        pylint_args = [str(temp_path), "--score=n", "--reports=n"]
        try:
            Run(pylint_args, reporter=reporter, exit=False)
        except TypeError:
            # Older pylint releases used `do_exit` instead of `exit`.
            Run(pylint_args, reporter=reporter, do_exit=False)
        raw = reporter_output.getvalue().strip()
        messages = json.loads(raw) if raw else []
    except Exception:
        messages = []
    finally:
        if temp_path is not None and (path is None or not path.exists()):
            try:
                temp_path.unlink()
            except Exception:
                pass

    issues: list[Issue] = []
    for message in messages:
        # Skip convention/info-level notes (e.g. missing docstrings) so they
        # do not turn every function into a refactor candidate.
        if message.get("type") not in {"warning", "error", "refactor"}:
            continue
        issues.append(
            Issue(
                line=int(message.get("line", 0)),
                message=str(message.get("message", "")),
                symbol=message.get("symbol"),
            )
        )

    return issues
