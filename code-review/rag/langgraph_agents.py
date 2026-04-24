"""
langgraph_agents.py — Multi-agent RAG orchestration using LangGraph.

Six specialized agents (Security, Architecture, Performance, Observability,
Database, Concurrency) are orchestrated as a LangGraph StateGraph:

    START → planner → fan-out to selected agents (parallel)
                    → aggregator (dedupe + rank) → END

Each agent has:
  • a domain-specific system prompt (persona + review criteria)
  • a fixed toolset (see AGENT_REGISTRY)
  • a deterministic context-gathering routine that invokes its tools,
    then a single LLM call that produces structured findings.

We do NOT rely on model-side tool calling (unreliable on local Ollama models).
Tools are invoked by Python orchestration code, and their outputs are
materialized into the agent's prompt. The LLM stays focused on review.
"""

from __future__ import annotations

import operator
from typing import Annotated, Callable, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, START, END

from rag.agent_tools import build_toolbox
from pipeline.deduplication import parse_findings


# ── Agent Registry ──────────────────────────────────────────────────────────

AGENT_REGISTRY: Dict[str, Dict] = {
    "security": {
        "persona": "AppSec Engineer",
        "focus": "SQL injection, auth bypass, hardcoded secrets, input validation, CVEs",
        "tools": ["SearchCode", "ReadFile", "QueryDependencyGraph", "ReadGoMod", "QueryRAG"],
        "rag_category": "security",
        "rule_ids": ["SEC-001", "SEC-002", "SEC-003"],
    },
    "architecture": {
        "persona": "Principal Architect",
        "focus": "service boundaries, API contracts, framework compliance, DI, imports",
        "tools": ["ParseAST", "QueryDependencyGraph", "ListDirectory", "QueryRAG"],
        "rag_category": None,
        "rule_ids": ["NAM-001", "NAM-002", "DOC-001"],
    },
    "performance": {
        "persona": "Performance Engineer",
        "focus": "N+1 queries, unbounded allocations, goroutine leaks, batch sizing",
        "tools": ["ExplainQuery", "ParseAST", "RunGolangCILint", "SearchCode", "QueryRAG"],
        "rag_category": "performance",
        "rule_ids": ["PERF-001"],
    },
    "observability": {
        "persona": "SRE / Platform Engineer",
        "focus": "logging completeness, trace spans, metrics naming, health endpoints",
        "tools": ["SearchCode", "ReadFile", "QueryRAG"],
        "rag_category": "logging",
        "rule_ids": ["LOG-001", "LOG-002"],
    },
    "database": {
        "persona": "DB Reliability Engineer",
        "focus": "transaction safety, pgx batch patterns, partition keys, migrations, indexes",
        "tools": ["QuerySchema", "ExplainQuery", "ReadFile", "ParseAST", "QueryRAG"],
        "rag_category": "security",  # SQL rules live under security in rules.json
        "rule_ids": ["SEC-002"],
    },
    "concurrency": {
        "persona": "Systems Engineer",
        "focus": "goroutine lifecycle, channel safety, mutex correctness, race conditions, errgroup",
        "tools": ["ParseAST", "SearchCode", "RunGoVet", "ReadFile", "QueryRAG"],
        "rag_category": "concurrency",
        "rule_ids": ["CONC-001", "CONC-002", "CTX-001", "CTX-002"],
    },
}


# ── Prompts ─────────────────────────────────────────────────────────────────

PLANNER_SYSTEM = """You are a triage planner for a Go code review. Given a Go
code snippet, decide which specialized review agents should inspect it.

Available agents: security, architecture, performance, observability,
database, concurrency.

Output STRICT JSON only (no prose, no fences):
{"agents": ["agent1", "agent2", ...]}

Pick 2 to 5 agents. Include an agent only if its focus plausibly applies."""


AGENT_SYSTEM_TEMPLATE = """You are a {persona} reviewing Go code.

FOCUS AREAS: {focus}

You are shown:
  (1) the code under review,
  (2) relevant coding-standard rules retrieved by RAG,
  (3) outputs from analysis tools you requested.

Report EVERY violation you find that falls within your focus area.
For each violation, emit EXACTLY this format:

### VIOLATION [RULE-ID] SEVERITY — Short Title
**File:** path/to/file.go:LINE
**Function:** functionName
**Issue:** Detailed description.
**Current code:**
```go
// the violating code
```
**Suggested fix:**
```go
// the corrected code
```

Severity: CRITICAL | HIGH | MEDIUM | LOW.
Prefer rule IDs from: {rule_ids}.
If nothing in your focus area applies, respond with: "No violations found."
Do NOT report violations outside your focus area — other agents cover those.
"""


# ── State ───────────────────────────────────────────────────────────────────


class ReviewState(TypedDict, total=False):
    code: str
    file_path: str
    repo_path: str
    selected_agents: List[str]
    findings: Annotated[List[Dict], operator.add]
    agent_trace: Annotated[List[str], operator.add]


# ── Orchestrator ────────────────────────────────────────────────────────────


class MultiAgentReviewer:
    """LangGraph-backed multi-agent code reviewer."""

    def __init__(
        self,
        rag_retriever,
        generate_fn: Callable[[str, str], str],
        repo_path: str,
        debug: bool = False,
    ):
        self.retriever = rag_retriever
        self.generate = generate_fn
        self.repo_path = repo_path
        self.debug = debug
        self.toolbox = build_toolbox(repo_path, rag_retriever)
        self.graph = self._build_graph()

    # ── Graph construction ─────────────────────────────────────────────────

    def _build_graph(self):
        g = StateGraph(ReviewState)
        g.add_node("planner", self._node_planner)
        g.add_node("aggregator", self._node_aggregator)
        for name in AGENT_REGISTRY:
            g.add_node(name, self._make_agent_node(name))

        g.add_edge(START, "planner")
        g.add_conditional_edges(
            "planner",
            lambda s: s.get("selected_agents", []) or ["aggregator"],
            {name: name for name in AGENT_REGISTRY} | {"aggregator": "aggregator"},
        )
        for name in AGENT_REGISTRY:
            g.add_edge(name, "aggregator")
        g.add_edge("aggregator", END)
        return g.compile()

    # ── Node: planner ──────────────────────────────────────────────────────

    def _node_planner(self, state: ReviewState) -> Dict:
        code = state["code"]
        user = f"```go\n{code[:1500]}\n```"
        try:
            raw = self.generate(PLANNER_SYSTEM, user)
        except Exception as e:
            if self.debug:
                print(f"[AGENTIC] planner failed: {e}; running all agents")
            return {"selected_agents": list(AGENT_REGISTRY.keys()), "agent_trace": ["planner: fallback"]}

        import json as _json
        import re as _re
        agents: List[str] = []
        m = _re.search(r"\{.*\}", raw, _re.DOTALL)
        if m:
            try:
                obj = _json.loads(m.group(0))
                for a in obj.get("agents", []):
                    a = str(a).strip().lower()
                    if a in AGENT_REGISTRY and a not in agents:
                        agents.append(a)
            except _json.JSONDecodeError:
                pass

        if not agents:
            # Safe default when planner returns nothing usable
            agents = ["security", "concurrency", "observability"]

        if self.debug:
            print(f"[AGENTIC] planner selected: {agents}")
        return {"selected_agents": agents, "agent_trace": [f"planner: {agents}"]}

    # ── Node: per-agent factory ────────────────────────────────────────────

    def _make_agent_node(self, name: str):
        spec = AGENT_REGISTRY[name]

        def _node(state: ReviewState) -> Dict:
            if name not in state.get("selected_agents", []):
                return {}  # skip (defensive; conditional edges already filter)

            code = state["code"]
            file_path = state.get("file_path", "")

            # 1) gather tool context deterministically
            tool_context = self._gather_tools(name, spec, code, file_path)

            # 2) RAG context
            rag_block = self.toolbox["QueryRAG"](
                f"{spec['focus']}:\n{code[:400]}",
                category=spec["rag_category"],
            )

            # 3) build user prompt and call LLM
            system = AGENT_SYSTEM_TEMPLATE.format(
                persona=spec["persona"],
                focus=spec["focus"],
                rule_ids=", ".join(spec["rule_ids"]) or "any applicable",
            )
            user = (
                f"## Relevant coding rules (RAG)\n{rag_block}\n\n"
                f"## Tool outputs\n{tool_context}\n\n"
                f"## File under review: {file_path}\n"
                f"```go\n{code}\n```\n"
            )

            try:
                raw = self.generate(system, user)
            except Exception as e:
                if self.debug:
                    print(f"[AGENTIC] {name} LLM call failed: {e}")
                return {"agent_trace": [f"{name}: error {e}"]}

            findings = parse_findings(raw)
            for f in findings:
                f.setdefault("category", name)
                if file_path:
                    f["file"] = file_path
                f["agent"] = name

            if self.debug:
                print(f"[AGENTIC] {name}: {len(findings)} finding(s)")
            return {
                "findings": findings,
                "agent_trace": [f"{name}: {len(findings)} finding(s)"],
            }

        return _node

    # ── Tool-gathering ─────────────────────────────────────────────────────

    def _gather_tools(self, agent_name: str, spec: Dict, code: str, file_path: str) -> str:
        """Invoke each tool this agent owns and collect concise outputs."""
        outputs: List[str] = []
        for tname in spec["tools"]:
            if tname == "QueryRAG":
                continue  # handled separately in the node
            try:
                fn = self.toolbox.get(tname)
                if fn is None:
                    continue

                if tname == "SearchCode":
                    # Pattern tailored to the agent's focus
                    pattern = _default_search_pattern(agent_name)
                    out = fn(pattern) if pattern else "[skipped: no default pattern]"
                elif tname in ("ReadFile", "ParseAST"):
                    out = fn(file_path) if file_path else "[skipped: no file_path]"
                elif tname == "ListDirectory":
                    out = fn("")
                elif tname == "ExplainQuery":
                    snippet = _extract_first_sql(code) or "[no sql literal found]"
                    out = fn(snippet)
                else:
                    out = fn()

                outputs.append(f"### {tname}\n{_truncate(out, 1500)}")
            except Exception as e:
                outputs.append(f"### {tname}\n[tool-error] {e}")
        return "\n\n".join(outputs) if outputs else "[no tool output]"

    # ── Node: aggregator ───────────────────────────────────────────────────

    def _node_aggregator(self, state: ReviewState) -> Dict:
        raw = state.get("findings", [])
        seen = set()
        merged: List[Dict] = []
        for f in raw:
            key = (f.get("rule_id", ""), f.get("file", ""), f.get("line_start", 0),
                   (f.get("title") or "")[:60])
            if key in seen:
                continue
            seen.add(key)
            merged.append(f)
        if self.debug:
            print(f"[AGENTIC] aggregator: {len(raw)} → {len(merged)} after dedupe")
        # Replace findings with deduped list. operator.add reducer means we
        # can't overwrite; so we return via a separate key the caller reads.
        return {"agent_trace": [f"aggregator: {len(merged)} unique finding(s)"]}

    # ── Public API ─────────────────────────────────────────────────────────

    def review_code(self, code: str, file_path: str = "") -> List[Dict]:
        """Run the multi-agent graph on one code chunk; return deduped findings."""
        init: ReviewState = {
            "code": code,
            "file_path": file_path,
            "repo_path": self.repo_path,
            "findings": [],
            "agent_trace": [],
        }
        final = self.graph.invoke(init)
        raw = final.get("findings", [])

        # Dedupe here (aggregator node can't mutate operator.add-reduced key).
        seen = set()
        out: List[Dict] = []
        for f in raw:
            key = (f.get("rule_id", ""), f.get("file", ""), f.get("line_start", 0))
            if key in seen:
                continue
            seen.add(key)
            out.append(f)
        return out

    def close(self) -> None:
        try:
            if self.retriever is not None:
                self.retriever.close()
        except Exception:
            pass


# ── Small helpers ───────────────────────────────────────────────────────────


def _truncate(text: str, n: int) -> str:
    if text is None:
        return ""
    return text if len(text) <= n else text[:n] + f"\n...(truncated {len(text) - n} chars)"


def _default_search_pattern(agent_name: str) -> Optional[str]:
    return {
        "security":      r'(password|secret|api[_-]?key|token|InsecureSkipVerify|fmt\.Sprintf.*SELECT)',
        "performance":   r'(append\(.*,.*\)|make\(\[\].*,\s*0\)|for .*range.*\bdb\.)',
        "observability": r'(fmt\.Print(f|ln)?\b|log\.Print(f|ln)?)',
        "concurrency":   r'(\bgo\s+func|\bgo\s+\w+\(|sync\.(Mutex|RWMutex|WaitGroup)|<-\s*\w|chan\s+\w)',
        "architecture":  r'^import\s|^package\s',
        "database":      r'(sql\.(Open|DB|Tx)|pgx\.|db\.(Query|Exec)|BEGIN;|COMMIT;)',
    }.get(agent_name)


def _extract_first_sql(code: str) -> Optional[str]:
    import re
    m = re.search(r'"([^"]*\b(SELECT|INSERT|UPDATE|DELETE)\b[^"]*)"', code, re.IGNORECASE)
    return m.group(1) if m else None
