"""
merge_findings.py
=================

Merge two architectural-review JSON reports (e.g. main run + recovery run)
by union over (file, function, rule_id). Later reports override earlier ones
for the same key.

Usage:
    python -m pipeline.merge_findings \
        --inputs results/architectural_review.json results/architectural_review_recovered.json \
        --output results/architectural_review_merged.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _key(f: Dict) -> Tuple[str, str, str]:
    return (str(f.get("file", "")).replace("\\", "/"), str(f.get("function", "")), str(f.get("rule_id", "")))


def merge(inputs: List[Path], output: Path) -> Dict:
    merged: Dict[Tuple[str, str, str], Dict] = {}
    repo_path = ""
    files_scanned = 0
    repo_files = 0
    handler_files = 0
    for p in inputs:
        doc = json.loads(p.read_text(encoding="utf-8"))
        repo_path = repo_path or doc.get("repo_path", "")
        files_scanned = max(files_scanned, int(doc.get("total_files_scanned", 0) or 0))
        repo_files = max(repo_files, int(doc.get("repo_files", 0) or 0))
        handler_files = max(handler_files, int(doc.get("handler_files", 0) or 0))
        for f in doc.get("findings", []):
            merged[_key(f)] = f

    findings = list(merged.values())
    counts: Dict[str, int] = {}
    summary_sev = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for f in findings:
        counts[f["rule_id"]] = counts.get(f["rule_id"], 0) + 1
        sev = str(f.get("severity", "")).upper()
        if sev in summary_sev:
            summary_sev[sev] += 1

    report = {
        "repo_path": repo_path,
        "total_files_scanned": files_scanned,
        "repo_files": repo_files,
        "handler_files": handler_files,
        "total_findings": len(findings),
        "findings_by_rule": counts,
        "findings": findings,
        "summary": {**summary_sev, "total": len(findings), "rules_violated": sorted(counts.keys())},
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[merge] inputs={len(inputs)} unique_findings={len(findings)} by_rule={counts}")
    print(f"[merge] wrote {output}")
    return report


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Merge architectural review JSON reports.")
    p.add_argument("--inputs", nargs="+", required=True, help="Input JSON paths (in priority order; later wins).")
    p.add_argument("--output", required=True, help="Output JSON path.")
    args = p.parse_args(argv)
    merge([Path(x).resolve() for x in args.inputs], Path(args.output).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
