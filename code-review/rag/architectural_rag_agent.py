"""
architectural_rag_agent.py — Agentic RAG Architectural Reviewer for Go Repositories

Enforces three structural rules using a pipeline of:
    1. Deterministic static analysis  (reused from architectural_analyzer.py)
       — walks all Go files, classifies them (repo / handler),
         extracts function bodies, and detects rule violations.
    2. RAG retrieval  (GoStandardsRetriever + Qdrant)
       — for each violation, retrieves the exact coding standard by rule_id
         plus semantically related context from the vector store.
    3. LLM code correction  (any generate_fn callable)
       — given the violating function source + retrieved standard,
         the LLM produces corrected Go code and a human-readable explanation.

Rules enforced
--------------
    REPO-001    — Repository function must contain exactly one database call.
    REPO-002    — Database call must not appear inside a for/range loop (N+1).
    HANDLER-001 — Handler function must not call more than one repository method.

Output format
-------------
Produces the same JSON shape as architectural_review.json (repo_path,
total_files_scanned, repo_files, handler_files, total_findings,
findings_by_rule, findings, summary) with three extra fields per finding:

    source_code          — full source of the violating function
    corrected_code       — LLM-generated corrected code
    llm_explanation      — LLM's explanation of the fix
    retrieved_standards  — the RAG-retrieved standard text used as context

Usage
-----
    from rag.architectural_rag_agent import ArchitecturalRAGAgent

    agent = ArchitecturalRAGAgent(
        repo_path      = "/path/to/go/repo",
        rag_retriever  = GoStandardsRetriever(db_path="rag/qdrant_db"),
        generate_fn    = my_llm_generate,   # callable(system: str, user: str) -> str
    )
    report = agent.analyze()
    print(report.to_json())

CLI
---
    python -m rag.architectural_rag_agent \\
        --repo   /path/to/repo \\
        --output results/rag_arch_review.json \\
        --model  ./models/go-reviewer
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# ── Ensure the package root is importable regardless of cwd ─────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Reuse all static-analysis machinery from architectural_analyzer.py ───────
# This keeps detection logic DRY — the RAG agent does not duplicate patterns.
from pipeline.architectural_analyzer import (
    GoFunction,
    ArchitecturalFinding,
    extract_functions,
    classify_file,
    _check_repo_function,
    _check_handler_function,
)

# ── RAG retriever ─────────────────────────────────────────────────────────────
from rag.retriever import GoStandardsRetriever


# ── Constants ─────────────────────────────────────────────────────────────────

# Max source-code characters sent to the LLM (prevents context overflow).
_MAX_SOURCE_CHARS = 6_000

# Max characters of retrieved standard text sent to the LLM.
_MAX_STANDARD_CHARS = 2_000

# Number of RAG documents to retrieve per finding.
_RAG_TOP_K = 3

# Category filter used for each rule when querying the vector store.
# Maps rule_id → category metadata value stored during indexing.
_RULE_CATEGORY: Dict[str, str] = {
    "REPO-001":    "architecture",
    "REPO-002":    "performance",
    "HANDLER-001": "architecture",
}


# ── LLM prompt templates ──────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert Go architect and code reviewer.
Your task is to fix a Go function that violates a specific architectural rule.
You will be given:
  1. The violated rule and its coding standard (retrieved from the standards database).
  2. The complete violating Go function source code.
  3. A description of exactly what went wrong.

You MUST respond in the following EXACT format — do not deviate:

### EXPLANATION
<concise explanation of why the code violates the rule and what the fix does>

### CORRECTED CODE
```go
<complete corrected Go function(s) — compilable, idiomatic Go>
```

Rules:
- Return COMPLETE, compilable Go code — not pseudocode or partial snippets.
- Preserve the original function signature, receiver, and package context.
- If the fix requires splitting a function, return ALL resulting functions.
- Do NOT add import statements (the caller handles imports).
- Do NOT wrap the corrected code block in anything other than ```go ... ```.
"""

_USER_TEMPLATE = """\
## Rule violated: {rule_id} — {title}
Severity: {severity}

## Coding Standard (from standards database)
{standard_text}

## Violation details
{issue}

## Offending lines detected by static analysis
{offending_lines}

## Full function source (lines {line_start}–{line_end} of {file})
```go
{source_code}
```

Now produce the EXPLANATION and CORRECTED CODE.
"""


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class RAGEnrichedFinding:
    """
    One architectural finding enriched with RAG-retrieved standard,
    the full violating source code, and LLM-generated corrected code.
    """
    # ── Fields from ArchitecturalFinding (static analysis) ──────────────────
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

    # ── RAG + LLM enrichment ─────────────────────────────────────────────────
    source_code: str = ""               # full function source extracted from file
    retrieved_standards: str = ""       # RAG-retrieved standard text used as context
    llm_explanation: str = ""           # LLM's explanation of what went wrong
    corrected_code: str = ""            # LLM-generated corrected Go code
    rag_docs_used: List[str] = field(default_factory=list)  # doc ids / titles retrieved

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ArchitecturalRAGReport:
    """Full report produced by the agentic RAG analyzer."""
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


# ── Source-code extraction ────────────────────────────────────────────────────

def _extract_source(abs_path: str, line_start: int, line_end: int) -> str:
    """
    Read the specified line range from a Go source file.

    Returns the source as a string, or an error placeholder on failure.
    The returned text is truncated to _MAX_SOURCE_CHARS to avoid overwhelming
    the LLM context window.
    """
    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        # Convert 1-based line numbers to 0-based slice indices.
        start_idx = max(0, line_start - 1)
        end_idx   = min(len(lines), line_end)
        source    = "".join(lines[start_idx:end_idx])
        if len(source) > _MAX_SOURCE_CHARS:
            source = source[:_MAX_SOURCE_CHARS] + f"\n// ... (truncated at {_MAX_SOURCE_CHARS} chars)"
        return source
    except OSError as exc:
        return f"// [error reading source: {exc}]"


# ── LLM output parsing ────────────────────────────────────────────────────────

_EXPLANATION_RE = re.compile(
    r"###\s*EXPLANATION\s*\n(.*?)(?=###|\Z)", re.DOTALL | re.IGNORECASE
)
_CORRECTED_CODE_RE = re.compile(
    r"###\s*CORRECTED CODE\s*\n```go\s*\n(.*?)```", re.DOTALL | re.IGNORECASE
)
# Fallback: any go code block in the response.
_ANY_GO_BLOCK_RE = re.compile(r"```go\s*\n(.*?)```", re.DOTALL)


def _parse_llm_response(raw: str) -> Tuple[str, str]:
    """
    Parse the LLM response and return (explanation, corrected_code).

    Falls back gracefully if the LLM deviates from the expected format:
    - Tries to find the EXPLANATION section.
    - Tries to find the CORRECTED CODE section with a ```go block.
    - Falls back to the first ```go block in the response.
    - If nothing matches, returns the raw response as the explanation.
    """
    explanation   = ""
    corrected_code = ""

    m_exp = _EXPLANATION_RE.search(raw)
    if m_exp:
        explanation = m_exp.group(1).strip()

    m_code = _CORRECTED_CODE_RE.search(raw)
    if m_code:
        corrected_code = m_code.group(1).strip()
    else:
        # Fallback: first ```go block anywhere in the response.
        m_fb = _ANY_GO_BLOCK_RE.search(raw)
        if m_fb:
            corrected_code = m_fb.group(1).strip()

    if not explanation and not corrected_code:
        # LLM returned something completely unexpected — store it raw.
        explanation = raw.strip()

    return explanation, corrected_code


# ── Main agent class ──────────────────────────────────────────────────────────

class ArchitecturalRAGAgent:
    """
    Agentic RAG architectural reviewer that combines:
      (a) deterministic static analysis  — finds violations precisely,
      (b) RAG retrieval                  — fetches the relevant coding standard,
      (c) LLM code correction            — generates corrected Go code.

    Parameters
    ----------
    repo_path : str
        Absolute path to the root of the Go repository to analyze.
    rag_retriever : GoStandardsRetriever
        An initialized retriever connected to the Qdrant vector store.
    generate_fn : Callable[[str, str], str]
        LLM generate function with signature (system_prompt, user_prompt) -> str.
        Can be an Ollama wrapper, Transformers pipeline, OpenAI client wrapper, etc.
    debug : bool, optional
        Print verbose progress information during analysis.
    """

    def __init__(
        self,
        repo_path: str,
        rag_retriever: GoStandardsRetriever,
        generate_fn: Callable[[str, str], str],
        debug: bool = False,
    ):
        self.repo_path  = os.path.abspath(repo_path)
        self.retriever  = rag_retriever
        self.generate   = generate_fn
        self.debug      = debug

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        if self.debug:
            print(f"[RAG-AGENT] {msg}")

    def _iter_go_files(self) -> List[str]:
        """Walk the repo and return all .go paths, skipping vendor/.git/testdata."""
        result: List[str] = []
        for dirpath, dirnames, filenames in os.walk(self.repo_path):
            dirnames[:] = [
                d for d in dirnames
                if d not in ("vendor", ".git", "node_modules", "testdata")
            ]
            for name in filenames:
                if name.endswith(".go"):
                    result.append(os.path.join(dirpath, name))
        return result

    def _rel(self, abs_path: str) -> str:
        """Return path relative to repo root (forward slashes)."""
        return os.path.relpath(abs_path, self.repo_path).replace("\\", "/")

    def _rag_retrieve(self, finding: ArchitecturalFinding) -> Tuple[str, List[str]]:
        """
        Retrieve the most relevant standard documents for a finding.

        Strategy:
          1. Query by rule_id + issue description (semantic similarity).
          2. Apply a category metadata filter to keep results focused.
          3. Combine retrieved texts into a single context block.

        Returns (formatted_standard_text, list_of_doc_titles).
        """
        query = (
            f"Go architectural rule {finding.rule_id}: {finding.title}. "
            f"{finding.issue[:300]}"
        )
        category = _RULE_CATEGORY.get(finding.rule_id)

        try:
            docs = self.retriever.retrieve(
                code_snippet=query,
                top_k=_RAG_TOP_K,
                category_filter=category,
            )
        except Exception as exc:
            self._log(f"RAG retrieval error for {finding.rule_id}: {exc}")
            docs = []

        if not docs:
            # Fallback: retry without category filter (vector store may not have
            # the category metadata field populated for all document types).
            try:
                docs = self.retriever.retrieve(code_snippet=query, top_k=_RAG_TOP_K)
            except Exception:
                docs = []

        if not docs:
            self._log(f"No RAG docs retrieved for {finding.rule_id} — using empty context")
            return "", []

        # Build formatted standard text block.
        blocks: List[str] = []
        titles: List[str] = []
        for i, doc in enumerate(docs, start=1):
            meta  = doc.metadata or {}
            title = meta.get("title") or meta.get("rule_id") or f"Standard #{i}"
            titles.append(title)
            blocks.append(f"[{i}] {title}\n{doc.page_content}")

        standard_text = "\n\n".join(blocks)
        if len(standard_text) > _MAX_STANDARD_CHARS:
            standard_text = standard_text[:_MAX_STANDARD_CHARS] + "\n... (truncated)"

        return standard_text, titles

    def _llm_correct(
        self,
        finding: ArchitecturalFinding,
        source_code: str,
        standard_text: str,
    ) -> Tuple[str, str]:
        """
        Call the LLM to produce corrected Go code and an explanation.

        Returns (explanation, corrected_code).
        """
        user = _USER_TEMPLATE.format(
            rule_id=finding.rule_id,
            title=finding.title,
            severity=finding.severity,
            standard_text=standard_text or "(no standard retrieved — use general Go best practices)",
            issue=finding.issue,
            offending_lines="\n".join(finding.offending_lines[:10]) or "(none)",
            line_start=finding.line_start,
            line_end=finding.line_end,
            file=finding.file,
            source_code=source_code or "(source not available)",
        )

        try:
            raw = self.generate(_SYSTEM_PROMPT, user)
            return _parse_llm_response(raw)
        except Exception as exc:
            self._log(f"LLM call failed for {finding.function} ({finding.rule_id}): {exc}")
            return f"LLM call failed: {exc}", ""

    def _enrich_finding(
        self,
        finding: ArchitecturalFinding,
        abs_path: str,
    ) -> RAGEnrichedFinding:
        """
        Full enrichment pipeline for one static-analysis finding:
          1. Extract source code.
          2. RAG retrieval.
          3. LLM correction.
        """
        self._log(
            f"Enriching {finding.rule_id} | {finding.function} | {finding.file}"
        )

        # 1 — Source extraction
        source_code = _extract_source(abs_path, finding.line_start, finding.line_end)

        # 2 — RAG retrieval
        standard_text, rag_titles = self._rag_retrieve(finding)

        # 3 — LLM correction
        explanation, corrected_code = self._llm_correct(finding, source_code, standard_text)

        return RAGEnrichedFinding(
            # Static analysis fields (pass-through)
            rule_id        = finding.rule_id,
            severity       = finding.severity,
            category       = finding.category,
            title          = finding.title,
            file           = finding.file,
            function       = finding.function,
            line_start     = finding.line_start,
            line_end       = finding.line_end,
            issue          = finding.issue,
            offending_lines= finding.offending_lines,
            suggested_fix  = finding.suggested_fix,
            # Enriched fields
            source_code          = source_code,
            retrieved_standards  = standard_text,
            llm_explanation      = explanation,
            corrected_code       = corrected_code,
            rag_docs_used        = rag_titles,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self) -> ArchitecturalRAGReport:
        """
        Perform the full agentic RAG analysis over the entire repository.

        Steps
        -----
        1. Walk all .go files in the repository.
        2. Classify each file as repo / handler / skip.
        3. Extract function bodies.
        4. Run deterministic static analysis (REPO-001, REPO-002, HANDLER-001).
        5. Enrich each finding with RAG + LLM (source code, standard, corrected code).
        6. Aggregate into an ArchitecturalRAGReport.

        Returns
        -------
        ArchitecturalRAGReport
            Contains the same top-level metadata as architectural_review.json
            plus enriched per-finding fields.
        """
        go_files        = self._iter_go_files()
        all_enriched:    List[RAGEnrichedFinding] = []
        repo_file_count = 0
        handler_file_count = 0

        self._log(f"Found {len(go_files)} Go files to scan in {self.repo_path}")

        for abs_path in go_files:
            try:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
                    source = fh.read()
            except OSError:
                continue

            file_rel = self._rel(abs_path)
            layer    = classify_file(abs_path, source)

            if layer is None:
                continue

            functions = extract_functions(source)

            # ── Static analysis ───────────────────────────────────────────
            raw_findings: List[ArchitecturalFinding] = []
            if layer == "repo":
                repo_file_count += 1
                for fn in functions:
                    raw_findings.extend(_check_repo_function(fn, file_rel))
            elif layer == "handler":
                handler_file_count += 1
                for fn in functions:
                    raw_findings.extend(_check_handler_function(fn, file_rel))

            if not raw_findings:
                continue

            self._log(
                f"  {file_rel} ({layer}): {len(raw_findings)} violation(s) — enriching…"
            )

            # ── RAG + LLM enrichment ──────────────────────────────────────
            for finding in raw_findings:
                enriched = self._enrich_finding(finding, abs_path)
                all_enriched.append(enriched)

        # ── Build report summary ──────────────────────────────────────────
        findings_by_rule: Dict[str, int] = {}
        severity_counts:  Dict[str, int] = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}

        for ef in all_enriched:
            findings_by_rule[ef.rule_id] = findings_by_rule.get(ef.rule_id, 0) + 1
            sev = ef.severity.upper()
            if sev in severity_counts:
                severity_counts[sev] += 1

        summary = {
            **severity_counts,
            "total":          len(all_enriched),
            "rules_violated": list(findings_by_rule.keys()),
        }

        self._log(
            f"Analysis complete: {len(all_enriched)} finding(s) across "
            f"{repo_file_count} repo file(s) and {handler_file_count} handler file(s)."
        )

        return ArchitecturalRAGReport(
            repo_path             = self.repo_path,
            total_files_scanned   = len(go_files),
            repo_files            = repo_file_count,
            handler_files         = handler_file_count,
            total_findings        = len(all_enriched),
            findings_by_rule      = findings_by_rule,
            findings              = [ef.to_dict() for ef in all_enriched],
            summary               = summary,
        )

    def analyze_file(self, file_path: str) -> List[RAGEnrichedFinding]:
        """
        Analyze a single .go file and return enriched findings.

        Useful for incremental / per-file review in CI pipelines.
        """
        abs_path = (
            file_path if os.path.isabs(file_path)
            else os.path.join(self.repo_path, file_path)
        )
        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
                source = fh.read()
        except OSError as exc:
            raise FileNotFoundError(f"Cannot read {abs_path}: {exc}") from exc

        file_rel = self._rel(abs_path)
        layer    = classify_file(abs_path, source)

        if layer is None:
            return []

        functions     = extract_functions(source)
        raw_findings: List[ArchitecturalFinding] = []

        if layer == "repo":
            for fn in functions:
                raw_findings.extend(_check_repo_function(fn, file_rel))
        elif layer == "handler":
            for fn in functions:
                raw_findings.extend(_check_handler_function(fn, file_rel))

        return [self._enrich_finding(f, abs_path) for f in raw_findings]

    def close(self) -> None:
        """Release the RAG retriever's Qdrant client connection."""
        try:
            self.retriever.close()
        except Exception:
            pass


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_generate_fn(model_path: str, device: str = "auto"):
    """
    Build a generate_fn callable from a local HuggingFace / PEFT model.
    Mirrors the loading pattern used in review_pipeline.py.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    try:
        from peft import PeftModel
        _peft_available = True
    except ImportError:
        _peft_available = False

    print(f"[INFO] Loading model from: {model_path}")

    quantization_config = None
    if torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    def generate_fn(system: str, user: str) -> str:
        if tokenizer.chat_template:
            messages = [
                {"role": "system",    "content": system},
                {"role": "user",      "content": user},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = f"### System:\n{system}\n\n### User:\n{user}\n\n### Assistant:\n"

        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    return generate_fn


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Agentic RAG architectural reviewer for Go repositories.\n"
            "Runs static analysis (REPO-001, REPO-002, HANDLER-001), then enriches\n"
            "each finding with RAG-retrieved standards and LLM-generated corrected code."
        )
    )
    parser.add_argument("--repo",   required=True,  help="Path to Go repository root")
    parser.add_argument("--output", default=None,   help="Write JSON report to this file (default: stdout)")
    parser.add_argument("--model",  default=None,   help="Path to local HuggingFace model for LLM correction")
    parser.add_argument("--db-path",default="rag/qdrant_db", help="Path to Qdrant vector store (default: rag/qdrant_db)")
    parser.add_argument("--debug",  action="store_true",    help="Print verbose progress information")
    args = parser.parse_args()

    # ── Build RAG retriever ──────────────────────────────────────────────────
    db_path = args.db_path
    if not os.path.isabs(db_path):
        # Resolve relative to the code-review package root.
        db_path = os.path.join(_ROOT, db_path)

    print(f"[INFO] Connecting to Qdrant vector store at: {db_path}")
    retriever = GoStandardsRetriever(db_path=db_path)

    # ── Build generate_fn ────────────────────────────────────────────────────
    if args.model:
        generate_fn = _build_generate_fn(args.model)
    else:
        print(
            "[WARN] No --model provided. LLM correction will be skipped.\n"
            "       Use --model /path/to/model to enable corrected-code generation.\n"
            "       Findings will still be produced (static analysis + RAG context only)."
        )
        def generate_fn(system: str, user: str) -> str:  # noqa: E306
            return (
                "### EXPLANATION\n"
                "No LLM model was provided. Run with --model to enable code correction.\n\n"
                "### CORRECTED CODE\n"
                "```go\n// No model — cannot generate corrected code.\n```"
            )

    # ── Run agent ─────────────────────────────────────────────────────────────
    agent = ArchitecturalRAGAgent(
        repo_path     = args.repo,
        rag_retriever = retriever,
        generate_fn   = generate_fn,
        debug         = args.debug,
    )

    try:
        report = agent.analyze()
        output = report.to_json()
    finally:
        agent.close()

    # ── Write output ─────────────────────────────────────────────────────────
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"[INFO] RAG architectural report saved to {args.output}")
    else:
        print(output)

    print(
        f"[INFO] Total findings: {report.total_findings}  "
        f"(CRITICAL: {report.summary.get('CRITICAL', 0)}, "
        f"HIGH: {report.summary.get('HIGH', 0)})"
    )


if __name__ == "__main__":
    main()
