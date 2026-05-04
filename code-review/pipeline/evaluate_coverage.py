"""
evaluate_coverage.py
====================

Compare the agentic RAG output (`architectural_review.json`) against the
deterministic pre-filter to surface:

  • MISSED   — functions where the deterministic scanner sees enough signal
               to trigger the rule, but the agent produced no finding.
  • EXTRA    — agent findings that the deterministic scanner did not flag
               (potential LLM hallucinations or wrap-around detections).
  • CONFIRMED — both agree.

The deterministic baseline reuses the SAME pre-filter functions the agent uses,
so this is a self-consistency check (the agent should ideally confirm every
candidate the pre-filter surfaces).

Usage
-----
    python -m pipeline.evaluate_coverage \\
        --repo "..\\post-data-management-back-end-development" \\
        --report results\\architectural_review.json \\
        --output results\\coverage_report.json \\
        --markdown results\\coverage_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from pipeline.architectural_rag_agent import (
    classify_file,
    extract_functions,
    candidates_for_rule,
    should_invoke_llm,
    walk_go_files,
)


@dataclass
class CoverageRow:
    rule_id: str
    file: str
    function: str
    line_start: int
    line_end: int
    deterministic_hits: int
    agent_detected: bool
    sample_hit_lines: List[str] = field(default_factory=list)


@dataclass
class CoverageReport:
    repo_path: str
    report_used: str
    summary: Dict
    missed: Dict[str, List[Dict]] = field(default_factory=dict)   # rule_id -> rows
    confirmed: Dict[str, List[Dict]] = field(default_factory=dict)
    extra: Dict[str, List[Dict]] = field(default_factory=dict)    # findings in agent but no det. signal

    def to_dict(self) -> Dict:
        return asdict(self)


# ─────────────────────────── Comparison logic ───────────────────────────────

def _key(file: str, function: str, rule_id: str) -> Tuple[str, str, str]:
    return (file.replace("\\", "/"), function, rule_id)


def evaluate(repo_path: Path, report_path: Path) -> CoverageReport:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    agent_keys: Set[Tuple[str, str, str]] = {
        _key(f["file"], f["function"], f["rule_id"]) for f in report.get("findings", [])
    }
    agent_findings_by_key = {
        _key(f["file"], f["function"], f["rule_id"]): f for f in report.get("findings", [])
    }

    rules_for_kind = {
        "repo": ["REPO-001", "REPO-002"],
        "handler": ["HANDLER-001"],
    }

    missed: Dict[str, List[Dict]] = {"REPO-001": [], "REPO-002": [], "HANDLER-001": []}
    confirmed: Dict[str, List[Dict]] = {"REPO-001": [], "REPO-002": [], "HANDLER-001": []}
    deterministic_keys: Set[Tuple[str, str, str]] = set()

    files_scanned = 0
    repo_files = 0
    handler_files = 0

    for path in walk_go_files(repo_path):
        try:
            src = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        rel = str(path.relative_to(repo_path)).replace("\\", "/")
        files_scanned += 1
        kind = classify_file(rel, src)
        if kind is None:
            continue
        if kind == "repo":
            repo_files += 1
        else:
            handler_files += 1

        functions = extract_functions(src)
        for fn in functions:
            for rule_id in rules_for_kind[kind]:
                hits = candidates_for_rule(rule_id, fn)
                if not should_invoke_llm(rule_id, hits):
                    continue
                k = _key(rel, fn.name, rule_id)
                deterministic_keys.add(k)
                row = CoverageRow(
                    rule_id=rule_id,
                    file=rel,
                    function=fn.name,
                    line_start=fn.start_line,
                    line_end=fn.end_line,
                    deterministic_hits=len(hits),
                    agent_detected=k in agent_keys,
                    sample_hit_lines=[f"line {ln}: {code}" for ln, code in hits[:5]],
                ).__dict__
                if k in agent_keys:
                    confirmed[rule_id].append(row)
                else:
                    missed[rule_id].append(row)

    # Anything in agent but NOT in deterministic baseline
    extra: Dict[str, List[Dict]] = {"REPO-001": [], "REPO-002": [], "HANDLER-001": []}
    for k, f in agent_findings_by_key.items():
        if k not in deterministic_keys:
            extra.setdefault(f["rule_id"], []).append({
                "rule_id": f["rule_id"],
                "file": f["file"],
                "function": f["function"],
                "line_start": f["line_start"],
                "line_end": f["line_end"],
                "offending_lines": f.get("offending_lines", []),
            })

    summary = {}
    for rid in ("REPO-001", "REPO-002", "HANDLER-001"):
        c = len(confirmed[rid])
        m = len(missed[rid])
        e = len(extra.get(rid, []))
        total_truth = c + m
        recall = (c / total_truth) if total_truth else 1.0
        summary[rid] = {
            "deterministic_candidates": total_truth,
            "agent_confirmed": c,
            "agent_missed": m,
            "agent_extra": e,
            "recall_pct": round(recall * 100, 2),
        }

    total_c = sum(s["agent_confirmed"] for s in summary.values())
    total_m = sum(s["agent_missed"] for s in summary.values())
    total_e = sum(s["agent_extra"] for s in summary.values())
    overall_recall = (total_c / (total_c + total_m)) if (total_c + total_m) else 1.0
    summary["OVERALL"] = {
        "deterministic_candidates": total_c + total_m,
        "agent_confirmed": total_c,
        "agent_missed": total_m,
        "agent_extra": total_e,
        "recall_pct": round(overall_recall * 100, 2),
        "files_scanned": files_scanned,
        "repo_files": repo_files,
        "handler_files": handler_files,
    }

    return CoverageReport(
        repo_path=str(repo_path.resolve()),
        report_used=str(report_path.resolve()),
        summary=summary,
        missed=missed,
        confirmed=confirmed,
        extra=extra,
    )


# ───────────────────────────── Markdown render ──────────────────────────────

def _md_table(rows: List[Dict], cols: List[Tuple[str, str]]) -> str:
    if not rows:
        return "_(none)_\n"
    header = "| " + " | ".join(label for _, label in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    lines = [header, sep]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(key, "")) for key, _ in cols) + " |")
    return "\n".join(lines) + "\n"


def write_markdown(rep: CoverageReport, path: Path) -> None:
    out: List[str] = []
    out.append(f"# Agentic-RAG Coverage Report\n")
    out.append(f"- Repo:   `{rep.repo_path}`")
    out.append(f"- Report: `{rep.report_used}`\n")

    out.append("## Summary\n")
    out.append("| Rule | Candidates | Confirmed | Missed | Extra | Recall % |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for rid in ("REPO-001", "REPO-002", "HANDLER-001", "OVERALL"):
        s = rep.summary[rid]
        out.append(
            f"| {rid} | {s['deterministic_candidates']} | {s['agent_confirmed']} "
            f"| {s['agent_missed']} | {s['agent_extra']} | {s['recall_pct']} |"
        )
    out.append("")

    cols_missed = [
        ("file", "File"),
        ("function", "Function"),
        ("line_start", "Start"),
        ("line_end", "End"),
        ("deterministic_hits", "Hits"),
    ]
    for rid in ("REPO-001", "REPO-002", "HANDLER-001"):
        out.append(f"## Missed by agent — {rid} ({len(rep.missed[rid])})\n")
        out.append(_md_table(rep.missed[rid], cols_missed))

        if rep.missed[rid]:
            out.append(f"### Sample evidence\n")
            for row in rep.missed[rid]:
                out.append(f"- **{row['file']}::{row['function']}** "
                           f"(lines {row['line_start']}-{row['line_end']})")
                for s in row["sample_hit_lines"]:
                    out.append(f"  - {s}")
            out.append("")

    out.append("## Extra findings (agent flagged, no deterministic signal)\n")
    cols_extra = [
        ("file", "File"),
        ("function", "Function"),
        ("rule_id", "Rule"),
        ("line_start", "Start"),
        ("line_end", "End"),
    ]
    flat_extra: List[Dict] = []
    for rid in ("REPO-001", "REPO-002", "HANDLER-001"):
        flat_extra.extend(rep.extra.get(rid, []))
    out.append(_md_table(flat_extra, cols_extra))

    path.write_text("\n".join(out), encoding="utf-8")


# ─────────────────────────────────── CLI ────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Coverage evaluator for the agentic-RAG reviewer.")
    p.add_argument("--repo", required=True, help="Path to the Go repository.")
    p.add_argument("--report", required=True, help="Path to the agent's architectural_review.json.")
    p.add_argument("--output", default="results/coverage_report.json")
    p.add_argument("--markdown", default="results/coverage_report.md")
    args = p.parse_args(argv)

    repo_path = Path(args.repo).resolve()
    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = (Path(__file__).resolve().parent.parent / report_path).resolve()

    if not repo_path.exists():
        print(f"[error] repo not found: {repo_path}", file=sys.stderr)
        return 2
    if not report_path.exists():
        print(f"[error] report not found: {report_path}", file=sys.stderr)
        return 2

    rep = evaluate(repo_path, report_path)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (Path(__file__).resolve().parent.parent / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rep.to_dict(), indent=2), encoding="utf-8")

    md_path = Path(args.markdown)
    if not md_path.is_absolute():
        md_path = (Path(__file__).resolve().parent.parent / md_path).resolve()
    write_markdown(rep, md_path)

    s = rep.summary["OVERALL"]
    print(
        f"[done] candidates={s['deterministic_candidates']} "
        f"confirmed={s['agent_confirmed']} missed={s['agent_missed']} "
        f"extra={s['agent_extra']} recall={s['recall_pct']}%"
    )
    print(f"[done] wrote {output_path}")
    print(f"[done] wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
