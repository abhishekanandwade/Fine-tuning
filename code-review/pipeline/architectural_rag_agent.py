"""
architectural_rag_agent.py
==========================

Agentic RAG architectural reviewer for Go repositories.

Detects exactly three rules:
    REPO-001   — Repository function must contain at most ONE database call.
    REPO-002   — DB call must not appear inside a for/range loop (N+1).
    HANDLER-001 — Handler must not call more than one repository/service method.

Pipeline (per Go file):

    classify_file ──► extract_functions ──► pre-filter (regex retrieval)
                                              │
                                              ▼
                              ┌──────── LangGraph agent ────────┐
                              │  retrieve_rule  →  analyze_llm  │
                              │           ▲             │       │
                              │           └── verify ◄──┘       │
                              └─────────────┬───────────────────┘
                                            ▼
                                        finding(s)

The "RAG" is two-tier:
  1. Lexical retrieval over `standards/architectural_rules.json` to fetch the
     rule definition + violation/correct examples for the candidate function.
  2. Code retrieval — the agent receives the function source plus the file's
     package declaration and may request an extra-context window (lines
     surrounding the function) before deciding.

Outputs `architectural_review.json` with the same schema as the deterministic
analyzer used to produce.

Usage
-----
    python -m pipeline.architectural_rag_agent \
        --repo "..\\post-data-management-back-end-development" \
        --output results\\architectural_review.json \
        --ollama-model qwen2.5-coder:7b

CLI flags:
    --repo               Path to the Go repository to analyze (required).
    --output             Output JSON path (default results/architectural_review.json).
    --rules              Path to rules JSON (default standards/architectural_rules.json).
    --ollama-model       Ollama model name (default qwen2.5-coder:7b).
    --ollama-url         Ollama base URL (default http://localhost:11434).
    --max-iterations     Max critic loops per finding (default 1).
    --limit-files        Stop after N qualifying files (debug aid).
    --filter-file        Substring filter on file path.
    --no-llm             Use only the deterministic pre-filter (skip LLM).
    --verbose            Verbose logging.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, TypedDict

import requests
from langgraph.graph import StateGraph, START, END


# ─────────────────────────── Constants & regexes ────────────────────────────

_REPO_DIR_KEYWORDS = frozenset({
    "repo", "repository", "repositories",
    "store", "storage", "stores", "dal", "dao",
})
_HANDLER_DIR_KEYWORDS = frozenset({
    "handler", "handlers", "http", "api",
    "controller", "controllers", "delivery",
})
_REPO_PACKAGE_NAMES = frozenset({"repo", "repository", "store", "storage", "dal", "dao", "postgres"})
_HANDLER_PACKAGE_NAMES = frozenset({"handler", "handlers", "http", "api", "controller", "delivery"})

# Broad DB-call indicator regex — used as a fast pre-filter to decide whether
# a function deserves an LLM round-trip.
_DB_HINT = re.compile(
    r"\b(?:"
    r"(?:[a-zA-Z_]\w*\.)?(?:db|DB|pool|Pool|conn|Conn|tx|Tx)\s*\."
    r"(?:Query|QueryRow|QueryContext|QueryRowContext|Exec|ExecContext|SendBatch|Begin|Prepare)"
    r"|dblib\.(?:Insert|Update|Delete|Select|SelectRows|SelectOne)"
    r"|psql\.(?:Insert|Update|Delete|Select)"
    r"|batch\.Queue"
    r"|br\.(?:Exec|Query|QueryRow)"
    r"|pgx\.RowToStruct"
    r")"
)

# Repository / service call indicator (used inside handlers).
_REPO_HINT = re.compile(
    r"\b(?:"
    r"pmh\.svc\."
    r"|h\.(?:svc|service|repo)\."
    r"|[a-zA-Z]+\.(?:svc|service|repo|Repo|Service|Repository)\."
    r"|[a-zA-Z]+[Rr]epo\."
    r"|[a-zA-Z]+[Ss]ervice\."
    r")[A-Z][a-zA-Z0-9_]*\s*\("
)

_FOR_LOOP_OPEN = re.compile(r"^\s*for\s+")

_FUNC_HEADER = re.compile(
    r"^func\s+"
    r"(?:\(\s*(?P<recv_var>\w+)\s+\*?(?P<recv_type>\w+)\s*\)\s+)?"
    r"(?P<name>\w+)\s*\("
)


# ─────────────────────────────── Data model ─────────────────────────────────

@dataclass
class GoFunction:
    name: str
    receiver_type: str
    start_line: int           # 1-based
    end_line: int             # 1-based
    body_lines: List[str]     # source lines, includes header line at index 0


@dataclass
class Finding:
    rule_id: str
    severity: str
    category: str
    title: str
    file: str
    function: str
    line_start: int
    line_end: int
    issue: str
    offending_lines: List[str] = field(default_factory=list)
    suggested_fix: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


# ─────────────────────────── Go source parsing ──────────────────────────────

def extract_package_name(source: str) -> str:
    m = re.search(r"^\s*package\s+(\w+)", source, re.MULTILINE)
    return m.group(1).lower() if m else ""


def classify_file(file_path: str, source: str) -> Optional[str]:
    parts = [p.lower() for p in Path(file_path).parts]
    if file_path.endswith("_test.go"):
        return None
    pkg = extract_package_name(source)
    if any(p in _REPO_DIR_KEYWORDS for p in parts) or pkg in _REPO_PACKAGE_NAMES:
        return "repo"
    if any(p in _HANDLER_DIR_KEYWORDS for p in parts) or pkg in _HANDLER_PACKAGE_NAMES:
        return "handler"
    return None


def extract_functions(source: str) -> List[GoFunction]:
    """Extract Go functions via brace-counting (no Go toolchain needed)."""
    lines = source.splitlines()
    n = len(lines)
    out: List[GoFunction] = []
    i = 0
    while i < n:
        m = _FUNC_HEADER.match(lines[i])
        if not m:
            i += 1
            continue
        start = i
        depth = 0
        in_block = False
        j = i
        end = n
        while j < n:
            raw = lines[j]
            s = raw
            if in_block:
                if "*/" in s:
                    s = s[s.index("*/") + 2:]
                    in_block = False
                else:
                    j += 1
                    continue
            if "/*" in s:
                pre = s[:s.index("/*")]
                post = ""
                if "*/" in s[s.index("/*"):]:
                    c = s.index("*/", s.index("/*"))
                    post = s[c + 2:]
                else:
                    in_block = True
                s = pre + post
            if "//" in s:
                s = s[:s.index("//")]
            s = re.sub(r"`[^`]*`", "", s)
            s = re.sub(r'"(?:[^"\\]|\\.)*"', '""', s)
            s = re.sub(r"'(?:[^'\\]|\\.)*'", "''", s)
            depth += s.count("{") - s.count("}")
            if depth > 0 or (depth == 0 and j == start):
                j += 1
                continue
            end = j + 1
            break
        out.append(GoFunction(
            name=m.group("name"),
            receiver_type=m.group("recv_type") or "",
            start_line=start + 1,
            end_line=end,
            body_lines=lines[start:end],
        ))
        i = end if end > start else start + 1
    return out


# ───────────── Deterministic pre-filter (RAG retrieval over code) ────────────

def db_lines_in_function(fn: GoFunction) -> List[Tuple[int, str]]:
    """Return (absolute_line_number, code) for lines that look like DB calls."""
    out: List[Tuple[int, str]] = []
    for idx, raw in enumerate(fn.body_lines, start=0):
        line_no = fn.start_line + idx
        # Skip pure comments
        stripped = raw.strip()
        if stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
            continue
        if _DB_HINT.search(raw):
            out.append((line_no, stripped))
    return out


def db_lines_inside_loops(fn: GoFunction) -> List[Tuple[int, str]]:
    """DB calls that appear inside a for/range loop within `fn`."""
    out: List[Tuple[int, str]] = []
    loop_stack: List[int] = []
    depth = 0
    in_block = False
    for idx, raw in enumerate(fn.body_lines, start=0):
        s = raw
        if in_block:
            if "*/" in s:
                s = s[s.index("*/") + 2:]
                in_block = False
            else:
                continue
        if "/*" in s:
            pre = s[:s.index("/*")]
            if "*/" in s[s.index("/*"):]:
                c = s.index("*/", s.index("/*"))
                s = pre + s[c + 2:]
            else:
                s = pre
                in_block = True
        if "//" in s:
            s = s[:s.index("//")]
        is_for_open = bool(_FOR_LOOP_OPEN.match(raw))
        prev_depth = depth
        depth += s.count("{") - s.count("}")
        if is_for_open and "{" in s:
            loop_stack.append(prev_depth + s.count("{"))
        loop_stack = [d for d in loop_stack if d <= depth]
        if loop_stack and _DB_HINT.search(raw):
            out.append((fn.start_line + idx, raw.strip()))
    return out


def repo_calls_in_handler(fn: GoFunction) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    for idx, raw in enumerate(fn.body_lines, start=0):
        stripped = raw.strip()
        if stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
            continue
        if _REPO_HINT.search(raw):
            out.append((fn.start_line + idx, stripped))
    return out


# ─────────────────────────── Ollama LLM wrapper ─────────────────────────────

def make_ollama_chat(
    model: str,
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0,
    seed: int = 42,
    num_predict: int = 384,
    timeout: int = 300,
) -> Callable[[str, str], str]:
    url = base_url.rstrip("/") + "/api/chat"

    def _call(system: str, user: str) -> str:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "seed": seed,
                "num_predict": num_predict,
            },
        }
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return (data.get("message") or {}).get("content", "") or ""

    return _call


def parse_json_lenient(text: str) -> Optional[Dict]:
    if not text:
        return None
    text = text.strip()
    # Extract first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    blob = text[start:end + 1]
    # Drop trailing commas
    blob = re.sub(r",(\s*[}\]])", r"\1", blob)
    try:
        return json.loads(blob)
    except Exception:
        return None


# ───────────────────────── LangGraph agent state ────────────────────────────

class AgentState(TypedDict, total=False):
    rule: Dict                       # rule definition (from rules.json)
    file_rel: str
    function_name: str
    line_start: int
    line_end: int
    function_source: str             # numbered source
    candidate_lines: List[Tuple[int, str]]   # deterministic hints
    package_name: str
    iterations: int
    llm_response: Dict               # parsed JSON from analyzer
    feedback: str                    # critic feedback for retry
    final: Optional[Dict]            # finding dict, or None if no violation
    _body_map: Dict                  # line_no -> raw source line (for deterministic critic)


# ─────────────────────────── Prompt templates ───────────────────────────────

_ANALYZER_SYS = """You are a strict Go architectural reviewer. You evaluate a single Go function against EXACTLY ONE rule and return a JSON verdict.

Output ONLY a JSON object (no prose, no markdown fences) with this schema:
{
  "violates": true|false,
  "offending_lines": ["line <N>: <verbatim code>", ...],
  "reasoning": "<1-2 sentences explaining the violation, or why not>"
}

Rules for output:
- offending_lines MUST cite line numbers that exist within the provided function range.
- offending_lines MUST quote the actual source line as it appears (you MAY trim leading whitespace).
- Only set violates=true when the rule is clearly broken.
- For REPO-001: violates=true requires >=2 distinct DB calls.
- For REPO-002: violates=true requires at least one DB call physically nested inside a for/range loop.
- For HANDLER-001: violates=true requires >=2 distinct repository/service method invocations from the handler body.
- Comments, string literals, and disabled (commented) code do NOT count as calls.
"""

_CRITIC_SYS = """You are a verification gate for an architectural reviewer's JSON output.

Given the rule, the function source, and the candidate verdict, return ONLY this JSON:
{ "ok": true|false, "feedback": "<short message>" }

Set ok=false if any of:
- offending_lines reference lines outside the function's range.
- offending_lines do not actually contain a database call (for REPO-001/REPO-002) or a repository/service call (for HANDLER-001).
- For REPO-002 the cited lines are not nested inside a for/range loop.
- The verdict claims violation but cites fewer than 2 DB calls (REPO-001) or 2 repo/service calls (HANDLER-001).
Otherwise ok=true.
"""


def _build_user_prompt(state: AgentState) -> str:
    rule = state["rule"]
    examples_block = (
        f"VIOLATION EXAMPLE:\n```go\n{rule.get('violation_example', '').strip()}\n```\n\n"
        f"CORRECT EXAMPLE:\n```go\n{rule.get('correct_example', '').strip()}\n```\n"
    )
    deterministic_hints = ""
    if state.get("candidate_lines"):
        hint_str = "\n".join(f"  line {ln}: {code}" for ln, code in state["candidate_lines"][:15])
        deterministic_hints = f"\n\nDETERMINISTIC SCAN HINTS (potential offenders, may include false positives):\n{hint_str}\n"
    feedback_block = ""
    if state.get("feedback"):
        feedback_block = f"\n\nPRIOR ATTEMPT FAILED VERIFICATION. Reviewer feedback:\n{state['feedback']}\nFix and re-emit JSON.\n"
    return (
        f"RULE: {rule['rule_id']} — {rule['title']}\n"
        f"DESCRIPTION: {rule['description']}\n\n"
        f"{examples_block}\n"
        f"FILE: {state['file_rel']} (package {state.get('package_name','')})\n"
        f"FUNCTION: {state['function_name']}  (lines {state['line_start']}-{state['line_end']})\n\n"
        f"FUNCTION SOURCE (line-numbered):\n```go\n{state['function_source']}\n```"
        f"{deterministic_hints}"
        f"{feedback_block}"
        f"\nReturn the JSON verdict now."
    )


def _build_critic_prompt(state: AgentState) -> str:
    rule = state["rule"]
    return (
        f"RULE: {rule['rule_id']} — {rule['title']}\n"
        f"DESCRIPTION: {rule['description']}\n\n"
        f"FUNCTION (lines {state['line_start']}-{state['line_end']}):\n```go\n{state['function_source']}\n```\n\n"
        f"CANDIDATE VERDICT JSON:\n{json.dumps(state.get('llm_response', {}), indent=2)}\n\n"
        f"Return JSON {{\"ok\": ..., \"feedback\": \"...\"}} now."
    )


# ────────────────────────────── Build graph ─────────────────────────────────

def build_agent(generate: Callable[[str, str], str], max_iterations: int = 1, verbose: bool = False, no_llm_veto: bool = False):

    def node_analyze(state: AgentState) -> AgentState:
        prompt = _build_user_prompt(state)
        raw = generate(_ANALYZER_SYS, prompt)
        parsed = parse_json_lenient(raw) or {"violates": False, "offending_lines": [], "reasoning": "[parse-failure]"}
        if verbose:
            print(f"    analyzer: violates={parsed.get('violates')} offending={len(parsed.get('offending_lines', []))}")
        return {"llm_response": parsed, "iterations": state.get("iterations", 0) + 1}

    def node_critic(state: AgentState) -> AgentState:
        """
        Deterministic critic: validates the analyzer's verdict programmatically
        instead of relying on a second LLM call. This is faster (no token cost)
        and avoids the false-negative loop we saw with an LLM critic that
        re-interpreted the rule incorrectly.

        Checks:
          1. Each offending_line cites a line number within [line_start, line_end].
          2. The cited source line actually contains the rule's pattern
             (DB call for REPO-001/REPO-002, repo/service call for HANDLER-001).
          3. For REPO-002, at least one cited line is nested inside a for/range loop.
          4. Minimum count: REPO-001 / HANDLER-001 require >=2 valid cites; REPO-002 >=1.
        """
        resp = state.get("llm_response") or {}
        rule_id = state["rule"]["rule_id"]
        if not resp.get("violates"):
            # --no-llm-veto: trust deterministic pre-filter when LLM disagrees.
            # Pre-filter already passed should_invoke_llm() quorum, so emit finding
            # using the validated candidate_lines as offenders.
            if no_llm_veto and state.get("candidate_lines"):
                hits = state["candidate_lines"]
                resp_recovered = {
                    "violates": True,
                    "offending_lines": [f"line {ln}: {code.strip()}" for ln, code in hits],
                    "reasoning": "[deterministic-recovered] LLM verdict overridden via --no-llm-veto; deterministic pre-filter quorum was met.",
                }
                if verbose:
                    print(f"    critic:   ok=True (no-llm-veto recovery, hits={len(hits)})")
                return {"llm_response": resp_recovered, "final": _build_finding_dict({**state, "llm_response": resp_recovered})}
            return {"final": None}
        line_start = state["line_start"]
        line_end = state["line_end"]
        offenders_raw = resp.get("offending_lines") or []

        # Build line_no -> source map from the function body (we have it via state).
        body_map = state.get("_body_map") or {}

        valid: List[str] = []
        for entry in offenders_raw:
            s = str(entry).strip()
            m = re.match(r"(?:line\s*)?(\d+)\s*[:\-]\s*(.+)$", s, re.IGNORECASE)
            if not m:
                continue
            ln = int(m.group(1))
            if ln < line_start or ln > line_end:
                continue
            actual = body_map.get(ln, m.group(2))
            # Validate the actual source line matches the rule's pattern.
            if rule_id in ("REPO-001", "REPO-002"):
                if not _DB_HINT.search(actual):
                    continue
            else:  # HANDLER-001
                if not _REPO_HINT.search(actual):
                    continue
            valid.append(f"line {ln}: {actual.strip()}")

        # Deduplicate by line number while preserving order.
        seen_lns = set()
        unique: List[str] = []
        for v in valid:
            ln = v.split(":", 1)[0]
            if ln not in seen_lns:
                seen_lns.add(ln)
                unique.append(v)

        # Apply rule-specific quorum.
        if rule_id == "REPO-002":
            # Require at least one valid line nested inside a loop.
            loop_lns = {ln for ln, _ in state.get("candidate_lines") or []}
            valid_in_loop = [v for v in unique if int(v.split(":", 1)[0].replace("line", "").strip()) in loop_lns]
            ok = len(valid_in_loop) >= 1
            unique = valid_in_loop or unique
        else:
            ok = len(unique) >= 2

        feedback = ""
        if not ok:
            feedback = (
                f"Verified offenders: {len(unique)}. "
                f"Need >=2 (REPO-001/HANDLER-001) or >=1 in-loop (REPO-002). "
                f"Re-cite ONLY lines from the function source (lines {line_start}-{line_end}) "
                f"that contain a real call. Use the format 'line <N>: <code>'."
            )

        if verbose:
            print(f"    critic:   ok={ok} verified={len(unique)} (deterministic)")

        if ok:
            # Replace LLM-emitted offenders with our validated list.
            resp["offending_lines"] = unique
            return {"llm_response": resp, "final": _build_finding_dict(state)}

        if state.get("iterations", 0) < max_iterations + 1:
            return {"feedback": feedback}
        return {"final": None}

    def route_after_critic(state: AgentState):
        if "final" in state:
            return END
        return "analyze"

    g = StateGraph(AgentState)
    g.add_node("analyze", node_analyze)
    g.add_node("critic", node_critic)
    g.add_edge(START, "analyze")
    g.add_edge("analyze", "critic")
    g.add_conditional_edges("critic", route_after_critic, {"analyze": "analyze", END: END})
    return g.compile()


def _build_finding_dict(state: AgentState) -> Dict:
    rule = state["rule"]
    resp = state.get("llm_response", {})
    offenders = [str(x) for x in (resp.get("offending_lines") or []) if x]
    issue = (
        f"{rule['title']} — {resp.get('reasoning', '').strip()}"
    ).strip()
    return Finding(
        rule_id=rule["rule_id"],
        severity=rule["severity"],
        category=rule["category"],
        title=rule["title"],
        file=state["file_rel"],
        function=state["function_name"],
        line_start=state["line_start"],
        line_end=state["line_end"],
        issue=issue,
        offending_lines=offenders,
        suggested_fix=rule.get("suggested_fix", ""),
    ).to_dict()


# ──────────────────────── Per-function orchestration ────────────────────────

def numbered_source(
    fn: GoFunction,
    hit_lines: Optional[List[int]] = None,
    max_lines: int = 90,
    window: int = 8,
) -> str:
    """
    Render the function body as a numbered listing.

    For long functions we keep the signature plus a context window around each
    deterministic hit line. Omitted regions are replaced with `... <N lines>`.
    This keeps the LLM prompt focused and dramatically reduces latency on
    >100-line functions while still showing the violation context.
    """
    total = len(fn.body_lines)
    if total <= max_lines or not hit_lines:
        out = []
        for idx, raw in enumerate(fn.body_lines, start=0):
            ln = fn.start_line + idx
            out.append(f"{ln:5d}  {raw}")
        return "\n".join(out)

    # Build set of body indices to keep: signature + windows around each hit.
    keep = set(range(0, min(3, total)))  # always keep first 3 lines (header)
    keep.add(total - 1)  # closing brace
    for ln in hit_lines:
        idx = ln - fn.start_line
        if 0 <= idx < total:
            for k in range(max(0, idx - window), min(total, idx + window + 1)):
                keep.add(k)

    out: List[str] = []
    prev_idx = -1
    for idx in sorted(keep):
        if prev_idx >= 0 and idx - prev_idx > 1:
            gap = idx - prev_idx - 1
            out.append(f"       ... <{gap} lines omitted>")
        ln = fn.start_line + idx
        out.append(f"{ln:5d}  {fn.body_lines[idx]}")
        prev_idx = idx
    return "\n".join(out)


def candidates_for_rule(rule_id: str, fn: GoFunction) -> List[Tuple[int, str]]:
    if rule_id == "REPO-001":
        return db_lines_in_function(fn)
    if rule_id == "REPO-002":
        return db_lines_inside_loops(fn)
    if rule_id == "HANDLER-001":
        return repo_calls_in_handler(fn)
    return []


def should_invoke_llm(rule_id: str, hits: List[Tuple[int, str]]) -> bool:
    """Skip LLM when deterministic pre-filter shows insufficient signal."""
    if rule_id in ("REPO-001", "HANDLER-001"):
        return len(hits) >= 2
    if rule_id == "REPO-002":
        return len(hits) >= 1
    return False


# ───────────────────────────── File walking ─────────────────────────────────

def walk_go_files(repo_path: Path):
    skip_dirs = {"vendor", "node_modules", ".git", "test_suite", "docs"}
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for fn in files:
            if fn.endswith(".go") and not fn.endswith("_test.go"):
                yield Path(root) / fn


# ─────────────────────────────── Main run ───────────────────────────────────

def run_review(
    repo_path: Path,
    rules_path: Path,
    output_path: Path,
    ollama_model: str,
    ollama_url: str,
    max_iterations: int,
    limit_files: Optional[int],
    filter_file: Optional[str],
    filter_functions: Optional[List[str]],
    no_llm: bool,
    verbose: bool,
    recover_from: Optional[Path] = None,
    no_llm_veto: bool = False,
) -> Dict:
    rules_doc = json.loads(rules_path.read_text(encoding="utf-8"))
    rules_by_id = {r["rule_id"]: r for r in rules_doc["rules"]}
    repo_rules = [rules_by_id["REPO-001"], rules_by_id["REPO-002"]]
    handler_rules = [rules_by_id["HANDLER-001"]]

    # --recover-from: load coverage report's missed list and build a (file, fn) -> {rule_ids} map.
    # When set, the agent runs ONLY on those exact triples (rule-aware, repo-agnostic).
    recovery_targets: Optional[Dict[Tuple[str, str], set]] = None
    if recover_from is not None:
        cov = json.loads(Path(recover_from).read_text(encoding="utf-8"))
        recovery_targets = {}
        for rid, rows in (cov.get("missed") or {}).items():
            for row in rows:
                key = (row["file"].replace("\\", "/"), row["function"])
                recovery_targets.setdefault(key, set()).add(rid)
        print(f"[recover] loaded {sum(len(v) for v in recovery_targets.values())} missed targets across {len(recovery_targets)} functions from {recover_from}")

    generate = None if no_llm else make_ollama_chat(ollama_model, ollama_url)
    agent = None if no_llm else build_agent(generate, max_iterations=max_iterations, verbose=verbose, no_llm_veto=no_llm_veto)

    findings: List[Dict] = []
    repo_files_count = 0
    handler_files_count = 0
    files_scanned = 0
    files_qualified = 0

    t0 = time.time()
    for path in walk_go_files(repo_path):
        rel = str(path.relative_to(repo_path)).replace("\\", "/")
        if filter_file and filter_file not in rel:
            continue
        try:
            src = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            if verbose:
                print(f"[skip] {rel}: {exc}")
            continue
        files_scanned += 1
        kind = classify_file(rel, src)
        if kind is None:
            continue
        if kind == "repo":
            repo_files_count += 1
            applicable = repo_rules
        else:
            handler_files_count += 1
            applicable = handler_rules
        files_qualified += 1
        if limit_files is not None and files_qualified > limit_files:
            break

        pkg = extract_package_name(src)
        functions = extract_functions(src)
        if verbose:
            print(f"[{kind}] {rel} -- {len(functions)} functions")

        for fn in functions:
            if filter_functions and fn.name not in filter_functions:
                continue
            target_rules: Optional[set] = None
            if recovery_targets is not None:
                target_rules = recovery_targets.get((rel, fn.name))
                if not target_rules:
                    continue
            for rule in applicable:
                if target_rules is not None and rule["rule_id"] not in target_rules:
                    continue
                hits = candidates_for_rule(rule["rule_id"], fn)
                if not should_invoke_llm(rule["rule_id"], hits):
                    continue
                if no_llm:
                    # Deterministic-only finding
                    findings.append(Finding(
                        rule_id=rule["rule_id"],
                        severity=rule["severity"],
                        category=rule["category"],
                        title=rule["title"],
                        file=rel,
                        function=fn.name,
                        line_start=fn.start_line,
                        line_end=fn.end_line,
                        issue=f"{rule['title']} (deterministic pre-filter; LLM disabled).",
                        offending_lines=[f"line {ln}: {code}" for ln, code in hits],
                        suggested_fix=rule.get("suggested_fix", ""),
                    ).to_dict())
                    continue

                state: AgentState = {
                    "rule": rule,
                    "file_rel": rel,
                    "function_name": fn.name,
                    "line_start": fn.start_line,
                    "line_end": fn.end_line,
                    "function_source": numbered_source(fn, hit_lines=[ln for ln, _ in hits]),
                    "candidate_lines": hits,
                    "package_name": pkg,
                    "iterations": 0,
                    "_body_map": {fn.start_line + i: raw for i, raw in enumerate(fn.body_lines)},
                }
                if verbose:
                    print(f"  -> {rule['rule_id']} {fn.name} (hits={len(hits)})")
                try:
                    out = agent.invoke(state)
                except Exception as exc:
                    if verbose:
                        print(f"    [graph-error] {exc}")
                    continue
                final = out.get("final")
                if final:
                    findings.append(final)

    # ── Build report (matches architectural_review.json schema) ──
    counts: Dict[str, int] = {}
    for f in findings:
        counts[f["rule_id"]] = counts.get(f["rule_id"], 0) + 1
    summary_sev = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for f in findings:
        sev = f.get("severity", "").upper()
        if sev in summary_sev:
            summary_sev[sev] += 1
    report = {
        "repo_path": str(repo_path.resolve()),
        "total_files_scanned": files_scanned,
        "repo_files": repo_files_count,
        "handler_files": handler_files_count,
        "total_findings": len(findings),
        "findings_by_rule": counts,
        "findings": findings,
        "summary": {
            **summary_sev,
            "total": len(findings),
            "rules_violated": sorted(counts.keys()),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    elapsed = time.time() - t0
    print(
        f"\n[done] files_scanned={files_scanned} repo={repo_files_count} "
        f"handler={handler_files_count} findings={len(findings)} "
        f"by_rule={counts} elapsed={elapsed:.1f}s"
    )
    print(f"[done] wrote {output_path}")
    return report


# ──────────────────────────────── CLI ───────────────────────────────────────

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Agentic-RAG architectural reviewer for Go repos.")
    p.add_argument("--repo", required=True, help="Path to the Go repository.")
    p.add_argument("--output", default="results/architectural_review.json")
    p.add_argument("--rules", default="standards/architectural_rules.json")
    p.add_argument("--ollama-model", default="qwen2.5-coder:7b")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--max-iterations", type=int, default=1)
    p.add_argument("--limit-files", type=int, default=None)
    p.add_argument("--filter-file", default=None)
    p.add_argument("--filter-functions", default=None,
                   help="Comma-separated list of function names to restrict analysis to.")
    p.add_argument("--no-llm", action="store_true",
                   help="Use only deterministic pre-filter (no LLM calls).")
    p.add_argument("--no-llm-veto", action="store_true",
                   help="When LLM says violates=False but deterministic quorum was met, "
                        "emit the finding anyway using the deterministic pre-filter hits. "
                        "Useful for recovery passes on functions the LLM second-guesses.")
    p.add_argument("--recover-from", default=None,
                   help="Path to a coverage_report.json. The agent will run ONLY on the "
                        "(file, function, rule) triples listed under 'missed' in that report. "
                        "Combine with --no-llm-veto for automatic gap recovery on any repo.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    repo_path = Path(args.repo).resolve()
    if not repo_path.exists():
        print(f"[error] repo path does not exist: {repo_path}", file=sys.stderr)
        return 2
    rules_path = Path(args.rules)
    if not rules_path.is_absolute():
        rules_path = (Path(__file__).resolve().parent.parent / rules_path).resolve()
    if not rules_path.exists():
        print(f"[error] rules file not found: {rules_path}", file=sys.stderr)
        return 2
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (Path(__file__).resolve().parent.parent / output_path).resolve()

    run_review(
        repo_path=repo_path,
        rules_path=rules_path,
        output_path=output_path,
        ollama_model=args.ollama_model,
        ollama_url=args.ollama_url,
        max_iterations=args.max_iterations,
        limit_files=args.limit_files,
        filter_file=args.filter_file,
        filter_functions=[s.strip() for s in args.filter_functions.split(",") if s.strip()] if args.filter_functions else None,
        no_llm=args.no_llm,
        verbose=args.verbose,
        recover_from=Path(args.recover_from).resolve() if args.recover_from else None,
        no_llm_veto=args.no_llm_veto,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
