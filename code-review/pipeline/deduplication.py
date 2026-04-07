"""
deduplication.py — Finding Deduplication and Ranking

Deduplicates review findings across chunks and ranks them by
severity and actionability.

Usage (as module):
    from pipeline.deduplication import deduplicate_findings, severity_rank
"""

import re
import hashlib
from typing import Dict, List, Optional
from collections import defaultdict


# ── Severity Ranking ─────────────────────────────────────────────────────────

SEVERITY_ORDER = {
    "CRITICAL": 0,
    "HIGH": 1,
    "MEDIUM": 2,
    "LOW": 3,
    "INFO": 4,
}


def severity_rank(severity: str) -> int:
    """
    Return numeric rank for severity (lower = more severe).
    Used for sorting findings with most critical first.
    """
    return SEVERITY_ORDER.get(severity.upper(), 5)


# ── Finding Parsing ──────────────────────────────────────────────────────────

def parse_findings(review_text: str) -> List[Dict]:
    """
    Parse structured findings from LLM review output.
    Extracts rule IDs, severity, descriptions, and code snippets.

    Expected format:
        ### VIOLATION [EH-001] HIGH — Title
        **File:** path/to/file.go:42
        **Issue:** Description of the issue...
        **Current code:**
        ```go
        // code
        ```
        **Suggested fix:**
        ```go
        // fixed code
        ```
    """
    findings = []

    # Split on VIOLATION markers
    violation_pattern = re.compile(
        r"###?\s*VIOLATION\s*\[([A-Z]+-\d+)\]\s*(CRITICAL|HIGH|MEDIUM|LOW|INFO)\s*(?:—|-)?\s*(.*?)(?=\n)",
        re.IGNORECASE,
    )

    matches = list(violation_pattern.finditer(review_text))

    for i, match in enumerate(matches):
        rule_id = match.group(1).upper()
        severity = match.group(2).upper()
        title = match.group(3).strip()

        # Extract the block between this match and the next
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(review_text)
        block = review_text[start:end]

        # Extract file and line
        file_match = re.search(r"\*\*File:\*\*\s*`?([^`\n]+):?(\d+)?`?", block)
        file_path = file_match.group(1).strip() if file_match else ""
        line_num = int(file_match.group(2)) if file_match and file_match.group(2) else 0

        # Extract function name
        func_match = re.search(r"\*\*Function:\*\*\s*`?([^`\n]+)`?", block)
        function_name = func_match.group(1).strip() if func_match else ""

        # Extract issue description
        issue_match = re.search(r"\*\*Issue:\*\*\s*(.*?)(?=\n\*\*|\n###|\Z)", block, re.DOTALL)
        description = issue_match.group(1).strip() if issue_match else title

        # Extract current code
        current_code = ""
        current_match = re.search(
            r"\*\*Current code[:\s]*\*\*\s*\n```\w*\n(.*?)```",
            block, re.DOTALL,
        )
        if current_match:
            current_code = current_match.group(1).strip()

        # Extract suggested fix
        suggested_fix = ""
        fix_match = re.search(
            r"\*\*Suggested fix[:\s]*\*\*\s*\n```\w*\n(.*?)```",
            block, re.DOTALL,
        )
        if fix_match:
            suggested_fix = fix_match.group(1).strip()

        # Determine effort
        effort = estimate_effort(current_code, suggested_fix)

        # Determine category from rule ID prefix
        category = rule_id.split("-")[0].lower()
        category_map = {
            "eh": "error_handling",
            "ctx": "context_usage",
            "log": "logging",
            "sec": "security",
            "nam": "naming",
            "conc": "concurrency",
            "test": "testing",
            "perf": "performance",
            "doc": "documentation",
        }
        category = category_map.get(category, "general")

        finding = {
            "rule_id": rule_id,
            "severity": severity,
            "category": category,
            "title": title,
            "file": file_path,
            "line_start": line_num,
            "line_end": line_num,
            "function": function_name,
            "description": description,
            "current_code": current_code,
            "suggested_fix": suggested_fix,
            "effort": effort,
            "auto_fixable": bool(suggested_fix),
        }
        findings.append(finding)

    return findings


def estimate_effort(current_code: str, suggested_fix: str) -> str:
    """Estimate the fix effort based on code diff size."""
    if not current_code and not suggested_fix:
        return "unknown"

    current_lines = len(current_code.split("\n")) if current_code else 0
    fix_lines = len(suggested_fix.split("\n")) if suggested_fix else 0

    total_change = abs(fix_lines - current_lines) + min(current_lines, fix_lines)

    if total_change <= 2:
        return "trivial"
    elif total_change <= 10:
        return "small"
    elif total_change <= 30:
        return "medium"
    else:
        return "large"


# ── Deduplication ────────────────────────────────────────────────────────────

def _finding_fingerprint(finding: Dict) -> str:
    """
    Generate a fingerprint for a finding to detect duplicates.
    Two findings are considered duplicates if they have the same
    rule_id + file + approximate line range.
    """
    key_parts = [
        finding.get("rule_id", ""),
        finding.get("file", ""),
        str(finding.get("line_start", 0) // 10),  # Group nearby lines
    ]

    if finding.get("current_code"):
        # Include a hash of the code to catch exact duplicates
        code_hash = hashlib.md5(
            finding["current_code"].encode("utf-8")
        ).hexdigest()[:8]
        key_parts.append(code_hash)

    return "|".join(key_parts)


def deduplicate_findings(findings: List[Dict]) -> List[Dict]:
    """
    Remove duplicate findings that appear across multiple chunks.

    When the same violation is found in overlapping chunks, keep the one
    with the most detailed description (longest) and highest severity.
    """
    fingerprint_groups: Dict[str, List[Dict]] = defaultdict(list)

    for finding in findings:
        fp = _finding_fingerprint(finding)
        fingerprint_groups[fp].append(finding)

    deduplicated = []
    for fp, group in fingerprint_groups.items():
        if len(group) == 1:
            deduplicated.append(group[0])
        else:
            # Keep the best finding from the group
            best = max(
                group,
                key=lambda f: (
                    -severity_rank(f.get("severity", "LOW")),  # Higher severity first
                    len(f.get("description", "")),              # Longer description
                    len(f.get("suggested_fix", "")),            # Has a fix
                ),
            )
            deduplicated.append(best)

    removed = len(findings) - len(deduplicated)
    if removed > 0:
        print(f"[INFO] Deduplication: {len(findings)} → {len(deduplicated)} (removed {removed} duplicates)")

    return deduplicated


def rank_findings(findings: List[Dict]) -> List[Dict]:
    """
    Sort findings by severity (critical first), then by file path and line.
    """
    return sorted(
        findings,
        key=lambda f: (
            severity_rank(f.get("severity", "LOW")),
            f.get("file", ""),
            f.get("line_start", 0),
        ),
    )


def group_findings_by_file(findings: List[Dict]) -> Dict[str, List[Dict]]:
    """Group findings by file path for file-level reporting."""
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for finding in findings:
        grouped[finding.get("file", "unknown")].append(finding)
    return dict(grouped)


def group_findings_by_category(findings: List[Dict]) -> Dict[str, List[Dict]]:
    """Group findings by rule category for category-level reporting."""
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for finding in findings:
        grouped[finding.get("category", "general")].append(finding)
    return dict(grouped)


def compute_summary(findings: List[Dict]) -> Dict:
    """Compute a summary of findings by severity."""
    summary = {
        "total": len(findings),
        "critical": 0,
        "high": 0,
        "medium": 0,
        "low": 0,
        "info": 0,
    }

    for finding in findings:
        sev = finding.get("severity", "LOW").lower()
        if sev in summary:
            summary[sev] += 1

    return summary
