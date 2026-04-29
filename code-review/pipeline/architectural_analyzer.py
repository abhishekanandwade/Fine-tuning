"""
architectural_analyzer.py — Deterministic Static Analysis for Repository/Handler Architecture

Enforces three structural rules without LLM inference:

    REPO-001  — A repository function must contain at most ONE database call.
    REPO-002  — A database call must not appear inside a for/range loop within
                a repository function.
    HANDLER-001 — A handler function must not call more than one repository method.

Classification strategy
-----------------------
* A file is classified as a **repository** file if:
    - its path contains a "repo", "repository", "store", or "storage" directory
      component, OR
    - its Go package name is "repo", "repository", "store", or "storage".
* A file is classified as a **handler** file if:
    - its path contains a "handler", "handlers", "http", "api", "controller",
      "controllers", or "delivery" directory component, OR
    - its Go package name is any of the above.

Detection approach
------------------
Function body extraction uses brace-counting on the raw source (no Go toolchain
required).  Pattern matching uses regexes compiled once at module load time.

This module is intentionally standalone — it does NOT depend on the LLM model,
RAG retriever, or Go toolchain (though it will use the AST helper if available).

Usage
-----
    from pipeline.architectural_analyzer import ArchitecturalAnalyzer

    analyzer = ArchitecturalAnalyzer(repo_path="/path/to/go/repo")
    report   = analyzer.analyze()
    print(report.to_json())
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── Constants ────────────────────────────────────────────────────────────────

# Directory name fragments that identify repository-layer files.
_REPO_DIR_KEYWORDS = frozenset({
    "repo", "repository", "repositories",
    "store", "storage", "stores",
    "dal", "dao", "db",
})

# Directory name fragments that identify handler-layer files.
_HANDLER_DIR_KEYWORDS = frozenset({
    "handler", "handlers",
    "http", "api",
    "controller", "controllers",
    "delivery", "transport",
    "server", "route", "routes",
    "middleware",
})

# Go package names that identify the repository layer.
_REPO_PACKAGE_NAMES = frozenset({
    "repo", "repository", "repositories",
    "store", "storage", "dal", "dao",
})

# Go package names that identify the handler layer.
_HANDLER_PACKAGE_NAMES = frozenset({
    "handler", "handlers", "http",
    "api", "controller", "controllers",
    "delivery", "transport", "server",
})

# ── DB-call detection patterns ───────────────────────────────────────────────
# Matches method calls on typical DB receiver field names (db, pool, conn, tx,
# DB, Pool, Conn, Tx, q, querier) followed by a DB-operation method.
# Handles both direct (r.db.Query) and chained (r.db.Pool.Query) receivers.
_DB_RECEIVER_PATTERN = re.compile(
    r"\b(?:r|s|u|q|repo|store|self)\s*\."        # common struct receiver vars
    r"(?:db|DB|pool|Pool|conn|Conn|tx|Tx|querier|q)\s*\."
    r"(?:"                                         # optional extra hop (e.g. .Pool.)
    r"(?:Pool|pool|Conn|conn|DB|db)\s*\.\s*"
    r")?"
    r"(?:Query|QueryRow|QueryContext|QueryRowContext"
    r"|Exec|ExecContext"
    r"|QueryRowx|QueryRowxContext"
    r"|Get|Select|NamedExec|NamedQuery"
    r"|SendBatch|BeginTx|Begin|BeginTxx"
    r"|Prepare|PrepareContext"
    r"|QueryxContext|Queryx|Scan)\b"
)

# Direct receiver calls: self.db.X(), or just db.Query(), pool.Exec() etc.
_DB_DIRECT_PATTERN = re.compile(
    r"(?<!\w)"
    r"(?:db|DB|pool|Pool|conn|Conn|sqlDB|pgDB|pgPool)\s*\."
    r"(?:Query|QueryRow|QueryContext|QueryRowContext"
    r"|Exec|ExecContext|QueryRowx|Get|Select"
    r"|SendBatch|BeginTx|Begin|Prepare)\b"
)

# pgx batch / pipeline patterns
_PGX_BATCH_PATTERN = re.compile(
    r"\bbatch\s*\.\s*(?:Queue|Send|SendBatch)\b"
    r"|\bconn\s*\.\s*SendBatch\b"
)

# Package-level DB wrapper functions used by custom ORM/helper libraries.
# Restricts to package aliases that contain DB-indicator substrings to
# avoid false positives (e.g. http.Get, json.Unmarshal are not DB calls).
_PKG_DB_FUNC_PATTERN = re.compile(
    r"\b(?:[a-z][a-zA-Z0-9_]*(?:db|sql|pg|store|repo|dal|orm|lib|query|DB|Sql|Pg))\s*\.\s*"
    r"(?:Insert|Update|Delete|Select|SelectOne|SelectRows"
    r"|Query|QueryRow|Exec|ExecContext|QueryContext"
    r"|Get|NamedExec|BulkInsert|BulkUpdate|Upsert)\s*\("
)

# Combined: any line containing a DB call
_ANY_DB_CALL = re.compile(
    r"(?:"
    + "|".join([
        _DB_RECEIVER_PATTERN.pattern,
        _DB_DIRECT_PATTERN.pattern,
        _PGX_BATCH_PATTERN.pattern,
        _PKG_DB_FUNC_PATTERN.pattern,
    ])
    + r")"
)

# ── Loop detection pattern ────────────────────────────────────────────────────
# Matches the opening line of a for/range statement (simplified; misses
# multi-line for-headers but covers the vast majority of real code).
_FOR_LOOP_OPEN = re.compile(r"^\s*for\s+")

# ── Repository call detection (inside handlers) ──────────────────────────────
# Identifies calls to repository methods from a handler.
# Strategy 1: receiver whose variable name or field ends with repo / store / dal
_REPO_RECEIVER_CALL = re.compile(
    r"\b(?:[a-zA-Z_]\w*\.)?"           # optional outer receiver
    r"(?:repo|Repo|repository|Repository|store|Store|storage|Storage"
    r"|dal|DAL|dao|DAO|user[Rr]epo|order[Rr]epo|[a-zA-Z]+[Rr]epo"
    r"|[a-zA-Z]+[Ss]tore|[a-zA-Z]+[Rr]epository)\s*\."
    r"[A-Z][a-zA-Z0-9_]*\s*\("         # method name starts with capital
)

# Strategy 2: CRUD-verb method call on any receiver starting with common prefixes
_CRUD_CALL = re.compile(
    r"\b\w+\s*\.\s*"
    r"(?:Get|Create|Update|Delete|Find|List|Save|Insert|Upsert|Fetch|Remove|Count|Exists)"
    r"[A-Z][a-zA-Z0-9_]*\s*\("
)

_ANY_REPO_CALL = re.compile(
    r"(?:"
    + "|".join([_REPO_RECEIVER_CALL.pattern, _CRUD_CALL.pattern])
    + r")"
)


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class ArchitecturalFinding:
    """One flagged violation produced by the architectural analyzer."""
    rule_id: str                  # REPO-001 | REPO-002 | HANDLER-001
    severity: str                 # HIGH | CRITICAL | MEDIUM
    category: str                 # "architecture" | "performance"
    title: str
    file: str                     # relative path inside repo
    function: str                 # function / method name
    line_start: int
    line_end: int
    issue: str                    # human-readable explanation
    offending_lines: List[str] = field(default_factory=list)  # relevant source lines
    suggested_fix: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ArchitecturalReport:
    """Full report produced by analyzing a Go repository."""
    repo_path: str = ""
    total_files_scanned: int = 0
    repo_files: int = 0
    handler_files: int = 0
    total_findings: int = 0
    findings_by_rule: Dict[str, int] = field(default_factory=dict)
    findings: List[Dict] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ── Function-body extraction ──────────────────────────────────────────────────

@dataclass
class GoFunction:
    """Lightweight representation of a parsed Go function."""
    name: str
    receiver_type: str            # empty string for top-level functions
    start_line: int               # 1-based, line of 'func' keyword
    end_line: int                 # 1-based, line of closing '}'
    body_lines: List[str] = field(default_factory=list)  # source lines (1-based positional)


_FUNC_HEADER = re.compile(
    r"^func\s+"
    r"(?:\(\s*(?P<recv_var>\w+)\s+\*?(?P<recv_type>\w+)\s*\)\s+)?"  # optional receiver
    r"(?P<name>\w+)\s*\("
)


def extract_functions(source: str) -> List[GoFunction]:
    """
    Extract all top-level function / method definitions from Go source text.

    Uses brace-counting rather than a full AST so it works without the Go
    toolchain.  Handles nested braces correctly (struct literals, closures).
    It does NOT handle string/comment literals that contain unbalanced braces,
    but these are exceedingly rare in production Go code.
    """
    lines = source.splitlines()
    functions: List[GoFunction] = []
    i = 0
    n = len(lines)

    while i < n:
        m = _FUNC_HEADER.match(lines[i])
        if m:
            func_name = m.group("name")
            recv_type = m.group("recv_type") or ""
            start_line = i + 1   # 1-based

            # Count braces to find the closing brace of this function.
            depth = 0
            j = i
            end_line = i + 1
            in_block_comment = False
            while j < n:
                raw_line = lines[j]

                # Track block comments /* ... */ to avoid counting braces inside them.
                # (Simplified: handles /* and */ on the same or different lines.)
                stripped = raw_line
                if in_block_comment:
                    if "*/" in stripped:
                        stripped = stripped[stripped.index("*/") + 2:]
                        in_block_comment = False
                    else:
                        j += 1
                        continue
                if "/*" in stripped:
                    before_comment = stripped[:stripped.index("/*")]
                    after_comment = ""
                    if "*/" in stripped[stripped.index("/*"):]:
                        after_start = stripped.index("/*")
                        close = stripped.index("*/", after_start)
                        after_comment = stripped[close + 2:]
                    else:
                        in_block_comment = True
                    stripped = before_comment + after_comment

                # Strip line comments
                if "//" in stripped:
                    stripped = stripped[:stripped.index("//")]

                # Strip string literals (very coarse — removes `...` and "..." and `...`)
                stripped = re.sub(r'`[^`]*`', '', stripped)
                stripped = re.sub(r'"(?:[^"\\]|\\.)*"', '""', stripped)
                stripped = re.sub(r"'(?:[^'\\]|\\.)*'", "''", stripped)

                depth += stripped.count("{") - stripped.count("}")

                if depth > 0 or (depth == 0 and j == i):
                    j += 1
                    continue
                # depth reached 0 — this is the closing brace
                end_line = j + 1   # 1-based
                break
            else:
                end_line = n

            functions.append(GoFunction(
                name=func_name,
                receiver_type=recv_type,
                start_line=start_line,
                end_line=end_line,
                body_lines=lines[i:end_line],  # zero-based slice → 1-based line numbers
            ))
            i = end_line  # skip past this function
        else:
            i += 1

    return functions


# ── File classification ───────────────────────────────────────────────────────

def _path_components(file_path: str) -> List[str]:
    """Return all directory / file name components of a path, lowercased."""
    return [p.lower() for p in Path(file_path).parts]


def _extract_package_name(source: str) -> str:
    m = re.search(r"^\s*package\s+(\w+)", source, re.MULTILINE)
    return m.group(1).lower() if m else ""


def classify_file(file_path: str, source: str) -> Optional[str]:
    """
    Return "repo", "handler", or None (unclassified / skip).

    Classification is based on directory path components and Go package name.
    The more-specific "repo" check takes priority over "handler" so that a
    file in `repo/handler_helpers.go` is treated as repo code.
    """
    parts = _path_components(file_path)
    pkg = _extract_package_name(source)

    # Skip test files
    if any(p.endswith("_test.go") for p in parts):
        return None
    if file_path.endswith("_test.go"):
        return None

    # Repository layer
    if any(p in _REPO_DIR_KEYWORDS for p in parts) or pkg in _REPO_PACKAGE_NAMES:
        return "repo"

    # Handler layer
    if any(p in _HANDLER_DIR_KEYWORDS for p in parts) or pkg in _HANDLER_PACKAGE_NAMES:
        return "handler"

    return None


# ── Loop-scope helper ─────────────────────────────────────────────────────────

def _db_calls_inside_loops(body_lines: List[str]) -> List[Tuple[int, str]]:
    """
    Return (line_number_in_body, stripped_line) pairs where a DB call occurs
    inside a for/range loop.

    `body_lines` is the slice of source lines that make up the function body
    (index 0 = start_line of the function).  Line numbers returned are
    relative to the start of the body (1-based).

    Approach: maintain a stack of loop-open brace depths so that we know
    whether we are currently inside a loop at each point in the code.
    """
    results: List[Tuple[int, str]] = []
    loop_depth_stack: List[int] = []  # brace depth at which each loop started
    current_depth = 0
    in_block_comment = False

    for idx, raw_line in enumerate(body_lines, start=1):
        stripped = raw_line

        # Handle block comments
        if in_block_comment:
            if "*/" in stripped:
                stripped = stripped[stripped.index("*/") + 2:]
                in_block_comment = False
            else:
                continue
        if "/*" in stripped:
            before = stripped[:stripped.index("/*")]
            if "*/" in stripped[stripped.index("/*"):]:
                close = stripped.index("*/", stripped.index("/*"))
                stripped = before + stripped[close + 2:]
            else:
                stripped = before
                in_block_comment = True

        # Strip line comments
        if "//" in stripped:
            stripped = stripped[:stripped.index("//")]

        is_loop_open = bool(_FOR_LOOP_OPEN.match(raw_line))

        # Count braces for nesting
        brace_delta = stripped.count("{") - stripped.count("}")
        prev_depth = current_depth
        current_depth += brace_delta

        # If this line opens a loop: push depth level AFTER this line's braces
        if is_loop_open and "{" in stripped:
            loop_depth_stack.append(prev_depth + stripped.count("{"))

        # Pop loop depth entries that have been exited (depth went below them)
        loop_depth_stack = [d for d in loop_depth_stack if d <= current_depth]

        # Check if we're inside any loop
        inside_loop = bool(loop_depth_stack)

        if inside_loop and _ANY_DB_CALL.search(raw_line):
            results.append((idx, raw_line.strip()))

    return results


# ── Rule implementations ──────────────────────────────────────────────────────

def _check_repo_function(
    fn: GoFunction,
    file_rel: str,
) -> List[ArchitecturalFinding]:
    """Apply REPO-001 and REPO-002 to a single repository function."""
    findings: List[ArchitecturalFinding] = []
    body = fn.body_lines

    # ── REPO-001: count total DB calls in the function ──
    db_call_lines: List[Tuple[int, str]] = []
    for idx, line in enumerate(body, start=1):
        if _ANY_DB_CALL.search(line):
            db_call_lines.append((fn.start_line + idx - 1, line.strip()))

    if len(db_call_lines) > 1:
        offending = [f"line {ln}: {code}" for ln, code in db_call_lines]
        findings.append(ArchitecturalFinding(
            rule_id="REPO-001",
            severity="HIGH",
            category="architecture",
            title="Multiple database calls in a single repository function",
            file=file_rel,
            function=fn.name,
            line_start=fn.start_line,
            line_end=fn.end_line,
            issue=(
                f"Repository function '{fn.name}' contains {len(db_call_lines)} database "
                f"calls. Each repository function must execute exactly one database "
                f"operation. Multiple DB calls can cause hidden N+1 problems, make "
                f"transactions unclear, and break the single-responsibility principle."
            ),
            offending_lines=offending,
            suggested_fix=(
                "Split this function into multiple single-responsibility repository "
                "functions, one per DB operation. If the operations are logically atomic, "
                "wrap them in a single database transaction and issue the calls through "
                "the transaction handle inside a dedicated helper."
            ),
        ))

    # ── REPO-002: DB call inside a for/range loop ──
    loop_db_hits = _db_calls_inside_loops(body)
    if loop_db_hits:
        offending = [f"line {fn.start_line + idx - 1}: {code}" for idx, code in loop_db_hits]
        findings.append(ArchitecturalFinding(
            rule_id="REPO-002",
            severity="CRITICAL",
            category="performance",
            title="Database call inside a loop (N+1 query pattern)",
            file=file_rel,
            function=fn.name,
            line_start=fn.start_line,
            line_end=fn.end_line,
            issue=(
                f"Repository function '{fn.name}' executes a database call inside a "
                f"for/range loop. This is the classic N+1 query anti-pattern: for N "
                f"iterations the code issues N separate round-trips to the database, "
                f"causing quadratic latency growth and connection-pool exhaustion."
            ),
            offending_lines=offending,
            suggested_fix=(
                "Collect all required identifiers first, then issue a single bulk query "
                "using IN ($1, $2, ...) or pgx SendBatch. Alternatively, use a JOIN or "
                "a CTE to retrieve all related data in one round-trip."
            ),
        ))

    return findings


def _check_handler_function(
    fn: GoFunction,
    file_rel: str,
) -> List[ArchitecturalFinding]:
    """Apply HANDLER-001 to a single handler function."""
    findings: List[ArchitecturalFinding] = []

    repo_call_lines: List[Tuple[int, str]] = []
    for idx, line in enumerate(fn.body_lines, start=1):
        if _ANY_REPO_CALL.search(line):
            # De-duplicate: skip lines that are just chaining (same line, multiple method calls)
            repo_call_lines.append((fn.start_line + idx - 1, line.strip()))

    # Deduplicate by actual line number (a single source line counts once)
    seen: set = set()
    unique_repo_calls: List[Tuple[int, str]] = []
    for ln, code in repo_call_lines:
        if ln not in seen:
            seen.add(ln)
            unique_repo_calls.append((ln, code))

    if len(unique_repo_calls) > 1:
        offending = [f"line {ln}: {code}" for ln, code in unique_repo_calls]
        findings.append(ArchitecturalFinding(
            rule_id="HANDLER-001",
            severity="HIGH",
            category="architecture",
            title="Handler function makes multiple repository calls",
            file=file_rel,
            function=fn.name,
            line_start=fn.start_line,
            line_end=fn.end_line,
            issue=(
                f"Handler function '{fn.name}' calls {len(unique_repo_calls)} repository "
                f"methods directly. Handlers should delegate business logic to a service "
                f"layer; the service layer is responsible for orchestrating multiple "
                f"repository calls. Keeping handlers thin prevents logic duplication, "
                f"improves testability, and maintains clean layer separation."
            ),
            offending_lines=offending,
            suggested_fix=(
                "Introduce a service / use-case layer (e.g. UserService.CreateWithProfile) "
                "that encapsulates the multi-repository orchestration. The handler calls "
                "exactly one service method; the service method calls as many repositories "
                "as needed, managing transactions internally."
            ),
        ))

    return findings


# ── Main analyzer class ───────────────────────────────────────────────────────

class ArchitecturalAnalyzer:
    """
    Deterministic, LLM-free static analyzer that enforces three architectural
    rules (REPO-001, REPO-002, HANDLER-001) across a Go repository.

    Parameters
    ----------
    repo_path : str
        Absolute path to the root of the Go repository to analyze.
    repo_dirs : list[str], optional
        Override the default set of directory keywords used to identify
        repository-layer files.
    handler_dirs : list[str], optional
        Override the default set of directory keywords used to identify
        handler-layer files.
    """

    def __init__(
        self,
        repo_path: str,
        repo_dirs: Optional[List[str]] = None,
        handler_dirs: Optional[List[str]] = None,
    ):
        self.repo_path = os.path.abspath(repo_path)
        if repo_dirs:
            global _REPO_DIR_KEYWORDS
            _REPO_DIR_KEYWORDS = frozenset(repo_dirs)
        if handler_dirs:
            global _HANDLER_DIR_KEYWORDS
            _HANDLER_DIR_KEYWORDS = frozenset(handler_dirs)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _iter_go_files(self) -> List[str]:
        """Walk the repo and return all .go file paths, skipping vendor/.git."""
        result: List[str] = []
        for dirpath, dirnames, filenames in os.walk(self.repo_path):
            # Prune vendor and .git from traversal
            dirnames[:] = [
                d for d in dirnames
                if d not in ("vendor", ".git", "node_modules", "testdata")
            ]
            for name in filenames:
                if name.endswith(".go"):
                    result.append(os.path.join(dirpath, name))
        return result

    def _rel(self, abs_path: str) -> str:
        """Return path relative to repo root (forward slashes for portability)."""
        return os.path.relpath(abs_path, self.repo_path).replace("\\", "/")

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self) -> ArchitecturalReport:
        """
        Analyze the entire repository and return an ArchitecturalReport.

        Steps:
            1. Walk .go files.
            2. Classify each file (repo / handler / skip).
            3. Extract function bodies from classified files.
            4. Apply rule checkers.
            5. Aggregate and return findings.
        """
        go_files = self._iter_go_files()

        all_findings: List[ArchitecturalFinding] = []
        repo_file_count = 0
        handler_file_count = 0

        for abs_path in go_files:
            try:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
                    source = fh.read()
            except OSError:
                continue

            file_rel = self._rel(abs_path)
            layer = classify_file(abs_path, source)

            if layer is None:
                continue

            functions = extract_functions(source)

            if layer == "repo":
                repo_file_count += 1
                for fn in functions:
                    all_findings.extend(_check_repo_function(fn, file_rel))

            elif layer == "handler":
                handler_file_count += 1
                for fn in functions:
                    all_findings.extend(_check_handler_function(fn, file_rel))

        # ── Build summary ──────────────────────────────────────────────────
        findings_by_rule: Dict[str, int] = {}
        severity_counts: Dict[str, int] = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for f in all_findings:
            findings_by_rule[f.rule_id] = findings_by_rule.get(f.rule_id, 0) + 1
            sev = f.severity.upper()
            if sev in severity_counts:
                severity_counts[sev] += 1

        summary = {
            **severity_counts,
            "total": len(all_findings),
            "rules_violated": list(findings_by_rule.keys()),
        }

        return ArchitecturalReport(
            repo_path=self.repo_path,
            total_files_scanned=len(go_files),
            repo_files=repo_file_count,
            handler_files=handler_file_count,
            total_findings=len(all_findings),
            findings_by_rule=findings_by_rule,
            findings=[f.to_dict() for f in all_findings],
            summary=summary,
        )

    def analyze_file(self, file_path: str) -> List[ArchitecturalFinding]:
        """
        Analyze a single .go file and return findings.

        Useful for incremental / per-file review in the main pipeline.
        """
        abs_path = file_path if os.path.isabs(file_path) else os.path.join(self.repo_path, file_path)

        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
                source = fh.read()
        except OSError as e:
            raise FileNotFoundError(f"Cannot read {abs_path}: {e}") from e

        file_rel = self._rel(abs_path)
        layer = classify_file(abs_path, source)

        if layer is None:
            return []

        functions = extract_functions(source)
        findings: List[ArchitecturalFinding] = []

        if layer == "repo":
            for fn in functions:
                findings.extend(_check_repo_function(fn, file_rel))
        elif layer == "handler":
            for fn in functions:
                findings.extend(_check_handler_function(fn, file_rel))

        return findings


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Architectural static analyzer for Go repositories (REPO-001/002, HANDLER-001)"
    )
    parser.add_argument("--repo", required=True, help="Path to Go repository root")
    parser.add_argument(
        "--output", default=None,
        help="Write JSON report to this file (default: print to stdout)"
    )
    parser.add_argument(
        "--repo-dirs", nargs="*", default=None,
        help="Override repository-layer directory keywords"
    )
    parser.add_argument(
        "--handler-dirs", nargs="*", default=None,
        help="Override handler-layer directory keywords"
    )
    args = parser.parse_args()

    analyzer = ArchitecturalAnalyzer(
        repo_path=args.repo,
        repo_dirs=args.repo_dirs,
        handler_dirs=args.handler_dirs,
    )
    report = analyzer.analyze()
    output = report.to_json()

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"[INFO] Architectural report saved to {args.output}")
        print(f"[INFO] Total findings: {report.total_findings}  "
              f"(CRITICAL: {report.summary.get('CRITICAL', 0)}, "
              f"HIGH: {report.summary.get('HIGH', 0)})")
    else:
        print(output)


if __name__ == "__main__":
    main()
