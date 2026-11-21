"""Find functions touched by a Pull Request.

1. `git diff --unified=0 base...head` -> changed line numbers per file (new side)
2. parse each changed Python file with `ast` -> functions whose span (incl. decorators)
   overlaps a changed line.
Other languages can be added later with tree-sitter; this is enough for v0.1.
"""

from __future__ import annotations

import ast
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
SUPPORTED_SUFFIXES = {".py"}


@dataclass
class ChangedFunction:
    path: str
    name: str
    start: int
    end: int
    code: str


def parse_diff(diff: str) -> dict[str, set[int]]:
    """Map file path -> set of added/modified line numbers on the new side."""
    changed: dict[str, set[int]] = {}
    current: str | None = None
    for line in diff.splitlines():
        if line.startswith("+++ "):
            target = line[4:].strip()
            current = None if target == "/dev/null" else target.removeprefix("b/")
            if current:
                changed.setdefault(current, set())
        elif current and (m := HUNK.match(line)):
            start, count = int(m.group(1)), int(m.group(2) or 1)
            changed[current].update(range(start, start + count))
    return {p: lines for p, lines in changed.items() if lines}


def git_diff(base: str, head: str, repo: Path = Path(".")) -> str:
    # refs come from PR metadata: refuse anything git could parse as an option
    for ref in (base, head):
        if not ref or ref.startswith("-"):
            raise ValueError(f"invalid git ref: {ref!r}")
    return subprocess.run(
        ["git", "diff", "--unified=0", "--no-color", f"{base}...{head}"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def functions_in_source(source: str, path: str = "<memory>") -> list[ChangedFunction]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    lines = source.splitlines()
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            # decorators carry the auth signal (@login_required), so include them
            start = min([d.lineno for d in node.decorator_list] + [node.lineno])
            end = node.end_lineno or node.lineno
            out.append(
                ChangedFunction(path, node.name, start, end, "\n".join(lines[start - 1 : end]))
            )
    return out


def changed_functions(
    diff: str, repo: Path = Path("."), max_lines: int = 400
) -> list[ChangedFunction]:
    result = []
    for path, lines in parse_diff(diff).items():
        file = repo / path
        if file.suffix not in SUPPORTED_SUFFIXES or not file.is_file():
            continue
        for fn in functions_in_source(file.read_text(encoding="utf-8", errors="replace"), path):
            if fn.end - fn.start > max_lines:
                continue  # generated / giant functions: skip, keeps CI under 60 s
            if any(fn.start <= ln <= fn.end for ln in lines):
                result.append(fn)
    # nested functions: keep the innermost-first order stable by file/line
    return sorted(result, key=lambda f: (f.path, f.start))
