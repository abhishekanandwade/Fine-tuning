"""
build_dataset.py — Dataset Builder for Go Code Review Training

Converts linter output (golangci-lint, gosec) and manual examples into
instruction-following JSONL format for fine-tuning an LLM code reviewer.

Usage:
    python dataset/build_dataset.py --repos-path /path/to/repos --output-dir dataset/processed
    python dataset/build_dataset.py --lint-json dataset/raw/lint_output.json --output-dir dataset/processed
    python dataset/build_dataset.py --manual-json dataset/raw/manual_examples.json --output-dir dataset/processed
"""

import argparse
import json
import os
import subprocess
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

# ── Constants ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert Go code reviewer enforcing company engineering standards. "
    "For each code snippet provided, identify all violations, explain why they "
    "violate standards, and provide the corrected code. Format your response with: "
    "VIOLATION [RULE-ID] SEVERITY, the issue description, the current code, and the "
    "suggested fix."
)

SEVERITY_MAP = {
    "error": "HIGH",
    "warning": "MEDIUM",
    "info": "LOW",
    "style": "LOW",
}

LINTER_TO_RULE_CATEGORY = {
    "errcheck": ("EH-002", "error_handling"),
    "goerr113": ("EH-001", "error_handling"),
    "wrapcheck": ("EH-001", "error_handling"),
    "staticcheck": ("MISC", "static_analysis"),
    "gosec": ("SEC", "security"),
    "revive": ("STYLE", "style"),
    "govet": ("MISC", "correctness"),
    "ineffassign": ("PERF", "performance"),
    "unused": ("MISC", "cleanup"),
    "gosimple": ("STYLE", "simplification"),
    "goconst": ("STYLE", "style"),
    "gocyclo": ("PERF", "complexity"),
    "misspell": ("DOC", "documentation"),
    "unparam": ("MISC", "cleanup"),
    "prealloc": ("PERF-001", "performance"),
    "bodyclose": ("SEC", "security"),
    "contextcheck": ("CTX-001", "context_usage"),
}


@dataclass
class LintFinding:
    """Represents a single linter finding."""
    file_path: str
    line: int
    column: int
    message: str
    linter: str
    severity: str
    code_snippet: str = ""
    suggested_fix: str = ""


@dataclass
class TrainingExample:
    """A single training example in chat format."""
    system: str
    user: str
    assistant: str


# ── Code Extraction ──────────────────────────────────────────────────────────

def extract_code_context(file_path: str, line: int, context_lines: int = 15) -> str:
    """
    Extract code around a specific line from a Go source file.
    Returns the function/block containing the line if possible.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except (FileNotFoundError, UnicodeDecodeError):
        return ""

    if line < 1 or line > len(lines):
        return ""

    # Try to find the enclosing function
    func_start = None
    func_end = None
    brace_count = 0

    # Walk backwards to find function start
    for i in range(line - 1, -1, -1):
        stripped = lines[i].strip()
        if stripped.startswith("func ") or re.match(r"func\s*\(", stripped):
            func_start = i
            break

    if func_start is not None:
        # Walk forward from func_start to find matching closing brace
        brace_count = 0
        for i in range(func_start, len(lines)):
            brace_count += lines[i].count("{") - lines[i].count("}")
            if brace_count == 0 and i > func_start:
                func_end = i + 1
                break

    if func_start is not None and func_end is not None:
        return "".join(lines[func_start:func_end])

    # Fallback: extract context_lines around the target line
    start = max(0, line - 1 - context_lines)
    end = min(len(lines), line + context_lines)
    return "".join(lines[start:end])


def extract_package_name(file_path: str) -> str:
    """Extract the package name from a Go source file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                stripped = raw_line.strip()
                if stripped.startswith("package "):
                    return stripped.split()[1]
    except (FileNotFoundError, UnicodeDecodeError):
        pass
    return "unknown"


# ── Linter Output Parsing ────────────────────────────────────────────────────

def parse_golangci_lint_json(json_path: str) -> List[LintFinding]:
    """
    Parse golangci-lint JSON output into LintFinding objects.
    Run: golangci-lint run ./... --out-format json > lint_output.json
    """
    findings = []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    issues = data.get("Issues", [])
    for issue in issues:
        pos = issue.get("Pos", {})
        file_path = pos.get("Filename", "")
        line = pos.get("Line", 0)
        column = pos.get("Column", 0)

        linter_name = issue.get("FromLinter", "unknown")
        severity = SEVERITY_MAP.get(issue.get("Severity", "warning"), "MEDIUM")

        code_snippet = extract_code_context(file_path, line)

        finding = LintFinding(
            file_path=file_path,
            line=line,
            column=column,
            message=issue.get("Text", ""),
            linter=linter_name,
            severity=severity,
            code_snippet=code_snippet,
        )
        findings.append(finding)

    return findings


def parse_gosec_json(json_path: str) -> List[LintFinding]:
    """
    Parse gosec JSON output into LintFinding objects.
    Run: gosec -fmt json ./... > security_output.json
    """
    findings = []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    issues = data.get("Issues", [])
    for issue in issues:
        file_path = issue.get("file", "")
        line = int(issue.get("line", "0"))
        column = int(issue.get("column", "0"))

        code_snippet = issue.get("code", "")
        if not code_snippet:
            code_snippet = extract_code_context(file_path, line)

        finding = LintFinding(
            file_path=file_path,
            line=line,
            column=column,
            message=issue.get("details", ""),
            linter="gosec",
            severity="HIGH",  # gosec findings are typically high severity
            code_snippet=code_snippet,
        )
        findings.append(finding)

    return findings


# ── Training Example Generation ──────────────────────────────────────────────

def finding_to_training_example(
    finding: LintFinding,
    rules: List[Dict],
) -> Optional[TrainingExample]:
    """
    Convert a single lint finding into a training example.
    Matches the finding to a known rule for the response template.
    """
    if not finding.code_snippet.strip():
        return None

    # Map linter to rule category
    rule_prefix, category = LINTER_TO_RULE_CATEGORY.get(
        finding.linter, ("MISC", "general")
    )

    # Try to find matching rule from our standards
    matched_rule = None
    for rule in rules:
        if rule["rule_id"].startswith(rule_prefix):
            matched_rule = rule
            break

    # Build user prompt
    rel_path = finding.file_path.replace("\\", "/")
    user_content = (
        f"Review the following Go code for standard violations:\n\n"
        f"**File:** {rel_path}\n\n"
        f"```go\n{finding.code_snippet.strip()}\n```"
    )

    # Build assistant response
    rule_id = matched_rule["rule_id"] if matched_rule else f"{rule_prefix}-000"
    title = matched_rule["title"] if matched_rule else finding.message

    assistant_content = (
        f"## Code Review Findings\n\n"
        f"### VIOLATION [{rule_id}] {finding.severity} — {title}\n"
        f"**File:** {rel_path}:{finding.line}\n"
        f"**Issue:** {finding.message}\n"
    )

    if matched_rule:
        assistant_content += (
            f"**Explanation:** {matched_rule['description']}\n\n"
            f"**Current code:**\n```go\n{matched_rule.get('violation_example', finding.code_snippet.strip())}\n```\n\n"
            f"**Suggested fix:**\n```go\n{matched_rule.get('correct_example', '// TODO: apply fix')}\n```\n\n"
        )
    else:
        assistant_content += (
            f"**Current code (line {finding.line}):**\n"
            f"```go\n{finding.code_snippet.strip()}\n```\n\n"
            f"**Action:** Review and fix the reported issue.\n\n"
        )

    assistant_content += f"**Summary:** 1 violation found — {finding.severity}."

    return TrainingExample(
        system=SYSTEM_PROMPT,
        user=user_content,
        assistant=assistant_content,
    )


def group_findings_by_file(
    findings: List[LintFinding],
) -> Dict[str, List[LintFinding]]:
    """Group lint findings by file path for multi-finding examples."""
    grouped: Dict[str, List[LintFinding]] = {}
    for f in findings:
        grouped.setdefault(f.file_path, []).append(f)
    return grouped


def findings_to_multi_example(
    file_path: str,
    file_findings: List[LintFinding],
    rules: List[Dict],
) -> Optional[TrainingExample]:
    """
    Convert multiple findings from the same file into a single training example.
    This teaches the model to find multiple issues in one pass.
    """
    if not file_findings:
        return None

    # Extract code for the whole file context (union of all snippets)
    code_snippets = set()
    for f in file_findings:
        if f.code_snippet.strip():
            code_snippets.add(f.code_snippet.strip())

    if not code_snippets:
        return None

    combined_code = "\n\n// ---\n\n".join(code_snippets)
    rel_path = file_path.replace("\\", "/")

    user_content = (
        f"Review the following Go code for standard violations:\n\n"
        f"**File:** {rel_path}\n\n"
        f"```go\n{combined_code}\n```"
    )

    # Build multi-finding response
    assistant_parts = ["## Code Review Findings\n"]
    severity_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}

    for i, finding in enumerate(file_findings, 1):
        rule_prefix, _ = LINTER_TO_RULE_CATEGORY.get(
            finding.linter, ("MISC", "general")
        )
        matched_rule = None
        for rule in rules:
            if rule["rule_id"].startswith(rule_prefix):
                matched_rule = rule
                break

        rule_id = matched_rule["rule_id"] if matched_rule else f"{rule_prefix}-000"
        title = matched_rule["title"] if matched_rule else finding.message

        assistant_parts.append(
            f"### VIOLATION [{rule_id}] {finding.severity} — {title}\n"
            f"**Line:** {finding.line}\n"
            f"**Issue:** {finding.message}\n"
        )
        severity_counts[finding.severity] = severity_counts.get(finding.severity, 0) + 1

    summary_parts = [
        f"{count} {sev.title()}" for sev, count in severity_counts.items() if count > 0
    ]
    assistant_parts.append(
        f"\n**Summary:** {len(file_findings)} violations found — {', '.join(summary_parts)}."
    )

    return TrainingExample(
        system=SYSTEM_PROMPT,
        user=user_content,
        assistant="\n".join(assistant_parts),
    )


# ── Manual Examples ──────────────────────────────────────────────────────────

def load_manual_examples(json_path: str) -> List[TrainingExample]:
    """
    Load manually crafted training examples from a JSON file.
    Expected format: list of {"code": "...", "review": "...", "file": "..."}
    """
    examples = []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        user_content = (
            f"Review the following Go code for standard violations:\n\n"
            f"**File:** {item.get('file', 'example.go')}\n\n"
            f"```go\n{item['code'].strip()}\n```"
        )

        examples.append(TrainingExample(
            system=SYSTEM_PROMPT,
            user=user_content,
            assistant=item["review"],
        ))

    return examples


# ── Linter Runner ────────────────────────────────────────────────────────────

def run_golangci_lint(repo_path: str, output_path: str) -> str:
    """
    Run golangci-lint on a Go repository and save JSON output.
    Requires golangci-lint to be installed.
    """
    cmd_candidates = [
        # Newer golangci-lint CLI (v2+)
        [
            "golangci-lint", "run", "./...",
            "--output.json.path", "stdout",
            "--timeout", "5m",
        ],
        # Legacy golangci-lint CLI
        [
            "golangci-lint", "run", "./...",
            "--out-format", "json",
            "--timeout", "5m",
        ],
    ]

    print(f"[INFO] Running golangci-lint on {repo_path}...")

    try:
        result = None
        for cmd in cmd_candidates:
            result = subprocess.run(
                cmd,
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )

            stderr = (result.stderr or "").strip().lower()
            unknown_flag = "unknown flag" in stderr
            if not unknown_flag:
                break

        if result is None:
            return ""

        # golangci-lint exits with 1 when issues are found, that's expected.
        # We still require machine-readable JSON in stdout.
        output = (result.stdout or "").strip()

        if not output:
            err = (result.stderr or "").strip()
            print(f"[ERROR] golangci-lint produced no JSON output for {repo_path}")
            if err:
                print(f"[ERROR] golangci-lint stderr: {err[:500]}")
            return ""

        # Validate JSON before writing, so downstream parser doesn't fail later.
        try:
            json.loads(output)
        except json.JSONDecodeError:
            # Some versions/plugins prepend text before JSON. Try to salvage
            # the first JSON object from stdout.
            start_idx = output.find("{")
            end_idx = output.rfind("}")
            salvaged = ""
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                salvaged = output[start_idx:end_idx + 1]
                try:
                    json.loads(salvaged)
                    output = salvaged
                except json.JSONDecodeError:
                    salvaged = ""

            if not salvaged:
                err = (result.stderr or "").strip()
                print(f"[ERROR] golangci-lint output is not valid JSON for {repo_path}")
                if err:
                    print(f"[ERROR] golangci-lint stderr: {err[:500]}")
                print(f"[ERROR] golangci-lint stdout sample: {output[:300]}")
                return ""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output)

        print(f"[INFO] Lint output saved to {output_path}")
        return output_path

    except FileNotFoundError:
        print("[ERROR] golangci-lint not found. Install: go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest")
        return ""
    except subprocess.TimeoutExpired:
        print(f"[ERROR] golangci-lint timed out on {repo_path}")
        return ""


def run_gosec(repo_path: str, output_path: str) -> str:
    """
    Run gosec on a Go repository and save JSON output.
    Requires gosec to be installed.
    """
    abs_output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(abs_output_path), exist_ok=True)

    cmd = ["gosec", "-fmt", "json", "-out", abs_output_path, "./..."]

    print(f"[INFO] Running gosec on {repo_path}...")

    try:
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if not os.path.exists(abs_output_path):
            err = (result.stderr or "").strip()
            print(f"[ERROR] gosec did not create output file: {abs_output_path}")
            if err:
                print(f"[ERROR] gosec stderr: {err[:500]}")
            return ""

        print(f"[INFO] Gosec output saved to {abs_output_path}")
        return abs_output_path

    except FileNotFoundError:
        print("[ERROR] gosec not found. Install: go install github.com/securego/gosec/v2/cmd/gosec@latest")
        return ""
    except subprocess.TimeoutExpired:
        print(f"[ERROR] gosec timed out on {repo_path}")
        return ""


# ── Dataset Building ─────────────────────────────────────────────────────────

def example_to_jsonl(example: TrainingExample) -> str:
    """Convert a TrainingExample to JSONL chat format."""
    obj = {
        "messages": [
            {"role": "system", "content": example.system},
            {"role": "user", "content": example.user},
            {"role": "assistant", "content": example.assistant},
        ]
    }
    return json.dumps(obj, ensure_ascii=False)


def split_dataset(
    examples: List[TrainingExample],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[TrainingExample], List[TrainingExample], List[TrainingExample]]:
    """Split examples into train/validation/test sets."""
    random.seed(seed)
    shuffled = examples.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return (
        shuffled[:train_end],
        shuffled[train_end:val_end],
        shuffled[val_end:],
    )


def write_jsonl(examples: List[TrainingExample], output_path: str) -> None:
    """Write training examples to a JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(example_to_jsonl(ex) + "\n")

    print(f"[INFO] Wrote {len(examples)} examples to {output_path}")


def discover_go_repositories(repos_path: str) -> List[Path]:
    """
    Discover Go repositories from a path.

    Supports:
    1) A single repository root (path itself contains go.mod)
    2) A directory containing multiple repos as direct children
    3) A directory tree containing nested repos (recursive fallback)
    """
    root = Path(repos_path)

    if not root.exists() or not root.is_dir():
        print(f"[ERROR] Invalid --repos-path: {repos_path}")
        return []

    # Case 1: path itself is a Go repository
    if (root / "go.mod").exists():
        return [root]

    # Case 2: direct child repositories
    direct_children = [
        d for d in root.iterdir()
        if d.is_dir() and (d / "go.mod").exists()
    ]
    if direct_children:
        return sorted(direct_children)

    # Case 3: recursive discovery
    nested_repo_dirs = {go_mod.parent for go_mod in root.rglob("go.mod")}
    return sorted(nested_repo_dirs)


def build_from_repos(
    repos_path: str,
    rules: List[Dict],
    raw_dir: str,
) -> List[TrainingExample]:
    """
    Run linters on all Go repos in a directory and convert findings
    to training examples.
    """

    print(f"[INFO] Scanning for Go repositories in {repos_path}...")
    
    examples = []
    repo_dirs = discover_go_repositories(repos_path)

    print(f"[INFO] Found {len(repo_dirs)} Go repositories")

    if not repo_dirs:
        print("[WARN] No Go repositories found. Ensure --repos-path points to a repo root or a directory containing repos with go.mod.")
        return examples

    for repo_dir in repo_dirs:
        repo_name = repo_dir.name
        print(f"\n{'='*60}")
        print(f"[INFO] Processing: {repo_name}")
        print(f"{'='*60}")

        # Run golangci-lint
        lint_output = os.path.join(raw_dir, f"{repo_name}_lint.json")
        if run_golangci_lint(str(repo_dir), lint_output):
            try:
                findings = parse_golangci_lint_json(lint_output)
                print(f"[INFO] Found {len(findings)} lint issues in {repo_name}")

                # Generate individual examples
                for finding in findings:
                    ex = finding_to_training_example(finding, rules)
                    if ex:
                        examples.append(ex)

                # Generate multi-finding examples (grouped by file)
                grouped = group_findings_by_file(findings)
                for fp, file_findings in grouped.items():
                    if len(file_findings) >= 2:
                        ex = findings_to_multi_example(fp, file_findings, rules)
                        if ex:
                            examples.append(ex)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[WARN] Failed to parse lint output for {repo_name}: {e}")

        # Run gosec
        sec_output = os.path.join(raw_dir, f"{repo_name}_gosec.json")
        if run_gosec(str(repo_dir), sec_output):
            try:
                sec_findings = parse_gosec_json(sec_output)
                print(f"[INFO] Found {len(sec_findings)} security issues in {repo_name}")
                for finding in sec_findings:
                    ex = finding_to_training_example(finding, rules)
                    if ex:
                        examples.append(ex)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[WARN] Failed to parse gosec output for {repo_name}: {e}")

    return examples


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build training dataset for Go code review model"
    )
    parser.add_argument(
        "--repos-path",
        help="Path to directory containing Go repositories",
    )
    parser.add_argument(
        "--lint-json",
        help="Path to existing golangci-lint JSON output",
    )
    parser.add_argument(
        "--gosec-json",
        help="Path to existing gosec JSON output",
    )
    parser.add_argument(
        "--manual-json",
        help="Path to manually crafted examples JSON",
    )
    parser.add_argument(
        "--rules-json",
        default="standards/rules.json",
        help="Path to coding standards rules JSON (default: standards/rules.json)",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset/processed",
        help="Output directory for train/val/test JSONL files",
    )
    parser.add_argument(
        "--raw-dir",
        default="dataset/raw",
        help="Directory to store raw linter outputs",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Proportion of data for training (default: 0.8)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Proportion of data for validation (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )

    args = parser.parse_args()

    # Load rules
    rules = []
    if os.path.exists(args.rules_json):
        with open(args.rules_json, "r", encoding="utf-8") as f:
            rules = json.load(f)
        print(f"[INFO] Loaded {len(rules)} rules from {args.rules_json}")
    else:
        print(f"[WARN] Rules file not found: {args.rules_json}")

    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    all_examples: List[TrainingExample] = []

    # Source 1: Run linters on repositories
    if args.repos_path:
        repo_examples = build_from_repos(args.repos_path, rules, args.raw_dir)
        all_examples.extend(repo_examples)
        print(f"[INFO] Generated {len(repo_examples)} examples from repositories")

    # Source 2: Parse existing lint JSON
    if args.lint_json:
        findings = parse_golangci_lint_json(args.lint_json)
        for f in findings:
            ex = finding_to_training_example(f, rules)
            if ex:
                all_examples.append(ex)
        print(f"[INFO] Generated {len(findings)} examples from lint JSON")

    # Source 3: Parse existing gosec JSON
    if args.gosec_json:
        findings = parse_gosec_json(args.gosec_json)
        for f in findings:
            ex = finding_to_training_example(f, rules)
            if ex:
                all_examples.append(ex)
        print(f"[INFO] Generated examples from gosec JSON")

    # Source 4: Load manual examples
    if args.manual_json:
        manual = load_manual_examples(args.manual_json)
        all_examples.extend(manual)
        print(f"[INFO] Loaded {len(manual)} manual examples")

    if not all_examples:
        print("[ERROR] No training examples generated. Provide --repos-path, --lint-json, --manual-json, or --gosec-json.")
        return

    # Deduplicate based on user content
    seen = set()
    unique_examples = []
    for ex in all_examples:
        key = ex.user[:200]
        if key not in seen:
            seen.add(key)
            unique_examples.append(ex)
    all_examples = unique_examples

    print(f"\n{'='*60}")
    print(f"[INFO] Total unique examples: {len(all_examples)}")

    # Split into train/val/test
    train, val, test = split_dataset(
        all_examples, args.train_ratio, args.val_ratio, args.seed
    )

    print(f"[INFO] Train: {len(train)} | Validation: {len(val)} | Test: {len(test)}")

    # Write output files
    write_jsonl(train, os.path.join(args.output_dir, "train.jsonl"))
    write_jsonl(val, os.path.join(args.output_dir, "validation.jsonl"))
    write_jsonl(test, os.path.join(args.output_dir, "test.jsonl"))

    print(f"\n[SUCCESS] Dataset built in {args.output_dir}/")


if __name__ == "__main__":
    main()
