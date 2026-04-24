"""
agent_tools.py — Tool functions used by the multi-agent LangGraph orchestrator.

Each tool is a plain Python callable (kept framework-agnostic) so agent nodes
can invoke them deterministically without relying on LLM tool-calling, which
is unreliable across local Ollama models. Tools degrade gracefully when the
external binary (go, golangci-lint) or resource is unavailable.
"""

from __future__ import annotations

import os
import re
import json
import shutil
import subprocess
from typing import Dict, List, Optional


# ── Helpers ──────────────────────────────────────────────────────────────────


def _safe_read(path: str, max_bytes: int = 200_000) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(max_bytes)
    except Exception as e:
        return f"[tool-error] could not read {path}: {e}"


def _iter_go_files(root: str) -> List[str]:
    out: List[str] = []
    for dirpath, _, files in os.walk(root):
        # Skip vendor / .git / node_modules
        if any(part in dirpath for part in (os.sep + "vendor", os.sep + ".git")):
            continue
        for name in files:
            if name.endswith(".go"):
                out.append(os.path.join(dirpath, name))
    return out


# ── Tools ────────────────────────────────────────────────────────────────────


def tool_search_code(pattern: str, repo_path: str, max_hits: int = 30) -> str:
    """Regex search across all .go files in repo. Returns matched lines with file:line."""
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return f"[tool-error] invalid regex: {e}"

    hits: List[str] = []
    for f in _iter_go_files(repo_path):
        try:
            with open(f, "r", encoding="utf-8", errors="replace") as fh:
                for lineno, line in enumerate(fh, 1):
                    if rx.search(line):
                        rel = os.path.relpath(f, repo_path)
                        hits.append(f"{rel}:{lineno}: {line.rstrip()}")
                        if len(hits) >= max_hits:
                            return "\n".join(hits) + f"\n...(truncated at {max_hits})"
        except Exception:
            continue
    return "\n".join(hits) if hits else "[no matches]"


def tool_read_file(path: str, repo_path: str) -> str:
    """Read a file relative to repo_path (or absolute)."""
    target = path if os.path.isabs(path) else os.path.join(repo_path, path)
    if not os.path.exists(target):
        return f"[tool-error] file not found: {path}"
    return _safe_read(target)


def tool_list_directory(rel_path: str, repo_path: str) -> str:
    """List files/dirs under repo_path/rel_path."""
    target = os.path.join(repo_path, rel_path) if rel_path else repo_path
    if not os.path.isdir(target):
        return f"[tool-error] not a directory: {rel_path}"
    entries = []
    for name in sorted(os.listdir(target)):
        full = os.path.join(target, name)
        entries.append(f"{name}{'/' if os.path.isdir(full) else ''}")
    return "\n".join(entries)


def tool_read_go_mod(repo_path: str) -> str:
    """Read go.mod at repo root (walks up if not found in root)."""
    for dirpath, _, files in os.walk(repo_path):
        if "go.mod" in files:
            return _safe_read(os.path.join(dirpath, "go.mod"))
    return "[tool-error] go.mod not found"


def tool_parse_ast(path: str, repo_path: str) -> str:
    """
    Lightweight AST-like summary via regex: imports, package, funcs, types,
    goroutine launches, channel ops. Avoids depending on `go` toolchain.
    """
    target = path if os.path.isabs(path) else os.path.join(repo_path, path)
    if not os.path.exists(target):
        return f"[tool-error] file not found: {path}"
    src = _safe_read(target)

    pkg = re.search(r"^package\s+(\w+)", src, re.MULTILINE)
    imports = re.findall(r'"([^"]+)"', re.search(
        r"import\s*\((.*?)\)", src, re.DOTALL).group(1)) if re.search(
        r"import\s*\(", src) else re.findall(r'^import\s+"([^"]+)"', src, re.MULTILINE)
    funcs = re.findall(r"^func\s+(?:\([^)]*\)\s+)?(\w+)\s*\(", src, re.MULTILINE)
    types = re.findall(r"^type\s+(\w+)\s+(struct|interface|\w+)", src, re.MULTILINE)
    goroutines = re.findall(r"^\s*go\s+\w", src, re.MULTILINE)
    chan_decls = re.findall(r"make\(chan\s+[^,)]+", src)

    return (
        f"package: {pkg.group(1) if pkg else '?'}\n"
        f"imports: {imports[:20]}\n"
        f"funcs ({len(funcs)}): {funcs[:25]}\n"
        f"types: {types[:20]}\n"
        f"goroutine_launches: {len(goroutines)}\n"
        f"channel_decls: {len(chan_decls)}"
    )


def tool_query_dependency_graph(repo_path: str) -> str:
    """Build a coarse import dependency map: file -> [imported modules]."""
    graph: Dict[str, List[str]] = {}
    for f in _iter_go_files(repo_path):
        src = _safe_read(f, max_bytes=20_000)
        block = re.search(r"import\s*\((.*?)\)", src, re.DOTALL)
        if block:
            imps = re.findall(r'"([^"]+)"', block.group(1))
        else:
            imps = re.findall(r'^import\s+"([^"]+)"', src, re.MULTILINE)
        graph[os.path.relpath(f, repo_path)] = imps
    # Compact textual summary
    lines = []
    for file, imps in list(graph.items())[:30]:
        lines.append(f"{file}: {imps[:10]}")
    return "\n".join(lines) if lines else "[no go files]"


def tool_run_golangci_lint(repo_path: str) -> str:
    """Run golangci-lint if installed; otherwise report unavailable."""
    if not shutil.which("golangci-lint"):
        return "[tool-unavailable] golangci-lint not installed"
    try:
        result = subprocess.run(
            ["golangci-lint", "run", "--out-format=line-number", "--timeout=60s"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=90,
        )
        out = (result.stdout or "") + (result.stderr or "")
        return out[:4000] if out.strip() else "[no issues reported]"
    except Exception as e:
        return f"[tool-error] golangci-lint failed: {e}"


def tool_run_go_vet(repo_path: str) -> str:
    """Run `go vet ./...` if go toolchain is installed."""
    if not shutil.which("go"):
        return "[tool-unavailable] go toolchain not installed"
    try:
        result = subprocess.run(
            ["go", "vet", "./..."],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=60,
        )
        out = (result.stdout or "") + (result.stderr or "")
        return out[:4000] if out.strip() else "[no issues reported]"
    except Exception as e:
        return f"[tool-error] go vet failed: {e}"


def tool_query_schema(repo_path: str) -> str:
    """Collect DDL from any .sql files under the repo."""
    parts: List[str] = []
    for dirpath, _, files in os.walk(repo_path):
        for name in files:
            if name.endswith(".sql"):
                rel = os.path.relpath(os.path.join(dirpath, name), repo_path)
                parts.append(f"-- {rel} --\n{_safe_read(os.path.join(dirpath, name), 5000)}")
    return "\n\n".join(parts)[:6000] if parts else "[no .sql files found]"


def tool_explain_query(query_snippet: str) -> str:
    """
    No live DB connection in this environment; produce a static heuristic
    analysis of the SQL snippet (flags N+1 patterns, missing WHERE, etc.).
    """
    q = query_snippet.strip()
    if not q:
        return "[tool-error] empty query"
    flags = []
    low = q.lower()
    if "select *" in low:
        flags.append("SELECT * — specify columns instead")
    if " where " not in low and low.startswith(("select", "update", "delete")):
        flags.append("missing WHERE clause — full-table scan")
    if "%s" in q or "fmt.sprintf" in low:
        flags.append("string-formatted query — SQL injection risk, use parameterized args")
    if " limit " not in low and low.startswith("select"):
        flags.append("no LIMIT — unbounded result set")
    return "analysis: " + ("; ".join(flags) if flags else "no obvious issues")


def tool_query_rag(retriever, query: str, category: Optional[str] = None, top_k: int = 4) -> str:
    """Call the underlying vector-store retriever and return formatted rules."""
    if retriever is None:
        return "[tool-unavailable] RAG retriever not initialized"
    try:
        docs = retriever.retrieve(query, top_k=top_k, category_filter=category)
        if not docs:
            docs = retriever.retrieve(query, top_k=top_k)
        return retriever.format_rules_for_prompt(docs)
    except Exception as e:
        return f"[tool-error] rag query failed: {e}"


# ── Registry ─────────────────────────────────────────────────────────────────


def build_toolbox(repo_path: str, rag_retriever) -> Dict[str, callable]:
    """Bind tools to the current repo/retriever context."""
    return {
        "SearchCode":            lambda pattern, max_hits=30: tool_search_code(pattern, repo_path, max_hits),
        "ReadFile":              lambda path: tool_read_file(path, repo_path),
        "ListDirectory":         lambda rel="": tool_list_directory(rel, repo_path),
        "ReadGoMod":             lambda: tool_read_go_mod(repo_path),
        "ParseAST":              lambda path: tool_parse_ast(path, repo_path),
        "QueryDependencyGraph":  lambda: tool_query_dependency_graph(repo_path),
        "RunGolangCILint":       lambda: tool_run_golangci_lint(repo_path),
        "RunGoVet":              lambda: tool_run_go_vet(repo_path),
        "QuerySchema":           lambda: tool_query_schema(repo_path),
        "ExplainQuery":          lambda q: tool_explain_query(q),
        "QueryRAG":              lambda q, category=None, top_k=4: tool_query_rag(rag_retriever, q, category, top_k),
    }
