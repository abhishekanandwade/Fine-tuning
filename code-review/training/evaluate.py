"""
evaluate.py — Evaluation Script for Go Code Review Model

Runs the fine-tuned model on the held-out test set and computes
precision, recall, F1, and per-category metrics.

Usage:
    python training/evaluate.py --model ./go-reviewer-final --test-file dataset/processed/test.jsonl
    python training/evaluate.py --model ./go-reviewer-final --test-file dataset/processed/test.jsonl --output-report evaluation_report.json
"""

import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class Finding:
    rule_id: str
    severity: str
    description: str


@dataclass
class EvalResult:
    example_id: int
    expected_findings: List[Finding]
    predicted_findings: List[Finding]
    true_positives: int
    false_positives: int
    false_negatives: int


# ── Parsing ──────────────────────────────────────────────────────────────────

def parse_findings_from_text(text: str) -> List[Finding]:
    """
    Parse structured findings from model output text.
    Looks for patterns like: VIOLATION [RULE-ID] SEVERITY — Title
    """
    findings = []

    # Pattern: ### VIOLATION [EH-001] HIGH — Title
    pattern = r"VIOLATION\s*\[([A-Z]+-\d+)\]\s*(CRITICAL|HIGH|MEDIUM|LOW)"
    matches = re.finditer(pattern, text, re.IGNORECASE)

    for match in matches:
        rule_id = match.group(1).upper()
        severity = match.group(2).upper()

        # Extract surrounding context as description
        start = match.start()
        end = min(len(text), start + 300)
        context = text[start:end]

        findings.append(Finding(
            rule_id=rule_id,
            severity=severity,
            description=context.strip(),
        ))

    return findings


def extract_rule_category(rule_id: str) -> str:
    """Extract the category prefix from a rule ID."""
    # EH-001 → EH, SEC-003 → SEC, CONC-002 → CONC
    parts = rule_id.split("-")
    return parts[0] if parts else "UNKNOWN"


# ── Model Inference ──────────────────────────────────────────────────────────

def load_model(model_path: str):
    """Load the fine-tuned model for evaluation."""
    print(f"[INFO] Loading model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    reviewer = pipeline(
        "text-generation",
        model=model_path,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    print("[INFO] Model loaded successfully")
    return reviewer, tokenizer


def generate_review(reviewer, tokenizer, messages: List[Dict]) -> str:
    """Generate a review given the system + user messages."""
    # Build prompt from messages (system + user only, no assistant)
    prompt_messages = [m for m in messages if m["role"] != "assistant"]

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            prompt = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = "\n".join(
                f"### {m['role'].title()}:\n{m['content']}" for m in prompt_messages
            )
    else:
        prompt = "\n".join(
            f"### {m['role'].title()}:\n{m['content']}" for m in prompt_messages
        )

    response = reviewer(
        prompt,
        max_new_tokens=1024,
        temperature=0.1,
        do_sample=True,
        repetition_penalty=1.1,
        return_full_text=False,
    )

    return response[0]["generated_text"]


# ── Evaluation Metrics ───────────────────────────────────────────────────────

def compute_finding_match(
    expected: List[Finding],
    predicted: List[Finding],
) -> Tuple[int, int, int]:
    """
    Compare expected vs predicted findings.
    A prediction matches if the rule_id category matches.
    Returns (true_positives, false_positives, false_negatives).
    """
    expected_categories = {extract_rule_category(f.rule_id) for f in expected}
    predicted_categories = {extract_rule_category(f.rule_id) for f in predicted}

    tp = len(expected_categories & predicted_categories)
    fp = len(predicted_categories - expected_categories)
    fn = len(expected_categories - predicted_categories)

    return tp, fp, fn


def compute_severity_accuracy(
    expected: List[Finding],
    predicted: List[Finding],
) -> float:
    """Compute severity accuracy for matched findings."""
    if not expected or not predicted:
        return 0.0

    correct = 0
    total = 0

    for exp in expected:
        exp_cat = extract_rule_category(exp.rule_id)
        for pred in predicted:
            pred_cat = extract_rule_category(pred.rule_id)
            if exp_cat == pred_cat:
                total += 1
                if exp.severity == pred.severity:
                    correct += 1
                break

    return correct / total if total > 0 else 0.0


def compute_aggregate_metrics(results: List[EvalResult]) -> Dict:
    """Compute aggregate precision, recall, F1 from all results."""
    total_tp = sum(r.true_positives for r in results)
    total_fp = sum(r.false_positives for r in results)
    total_fn = sum(r.false_negatives for r in results)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "total_examples": len(results),
        "total_true_positives": total_tp,
        "total_false_positives": total_fp,
        "total_false_negatives": total_fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
    }


def compute_per_category_metrics(results: List[EvalResult]) -> Dict:
    """Compute per-category (EH, SEC, etc.) precision/recall."""
    category_stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )

    for result in results:
        expected_cats = {extract_rule_category(f.rule_id) for f in result.expected_findings}
        predicted_cats = {extract_rule_category(f.rule_id) for f in result.predicted_findings}

        for cat in expected_cats & predicted_cats:
            category_stats[cat]["tp"] += 1
        for cat in predicted_cats - expected_cats:
            category_stats[cat]["fp"] += 1
        for cat in expected_cats - predicted_cats:
            category_stats[cat]["fn"] += 1

    per_category = {}
    for cat, stats in sorted(category_stats.items()):
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_category[cat] = {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
        }

    return per_category


# ── Main Evaluation Loop ─────────────────────────────────────────────────────

def evaluate(
    model_path: str,
    test_file: str,
    max_examples: Optional[int] = None,
) -> Dict:
    """
    Run evaluation on test set and compute all metrics.
    """
    # Load model
    reviewer, tokenizer = load_model(model_path)

    # Load test data
    print(f"[INFO] Loading test data from {test_file}...")
    test_examples = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_examples.append(json.loads(line))

    if max_examples:
        test_examples = test_examples[:max_examples]

    print(f"[INFO] Evaluating on {len(test_examples)} examples...")

    results: List[EvalResult] = []
    severity_accuracies: List[float] = []

    for i, example in enumerate(test_examples):
        messages = example["messages"]

        # Extract expected findings from ground truth (assistant message)
        expected_text = next(
            (m["content"] for m in messages if m["role"] == "assistant"), ""
        )
        expected_findings = parse_findings_from_text(expected_text)

        # Generate model prediction
        print(f"  [{i+1}/{len(test_examples)}] Generating review...")
        predicted_text = generate_review(reviewer, tokenizer, messages)
        predicted_findings = parse_findings_from_text(predicted_text)

        # Compute metrics for this example
        tp, fp, fn = compute_finding_match(expected_findings, predicted_findings)
        sev_acc = compute_severity_accuracy(expected_findings, predicted_findings)

        result = EvalResult(
            example_id=i,
            expected_findings=expected_findings,
            predicted_findings=predicted_findings,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
        )
        results.append(result)
        severity_accuracies.append(sev_acc)

        # Print per-example summary
        print(
            f"    Expected: {len(expected_findings)} | "
            f"Predicted: {len(predicted_findings)} | "
            f"TP={tp} FP={fp} FN={fn}"
        )

    # Aggregate metrics
    aggregate = compute_aggregate_metrics(results)
    per_category = compute_per_category_metrics(results)
    avg_severity_acc = (
        sum(severity_accuracies) / len(severity_accuracies) if severity_accuracies else 0.0
    )

    report = {
        "model": model_path,
        "test_file": test_file,
        "num_examples": len(results),
        "aggregate_metrics": aggregate,
        "severity_accuracy": round(avg_severity_acc, 4),
        "per_category_metrics": per_category,
        "target_thresholds": {
            "precision": "> 0.85",
            "recall": "> 0.70",
            "false_positive_rate": "< 0.15",
            "severity_accuracy": "> 0.75",
        },
    }

    return report


def print_report(report: Dict) -> None:
    """Print a formatted evaluation report."""
    print("\n" + "=" * 70)
    print("                    EVALUATION REPORT")
    print("=" * 70)

    agg = report["aggregate_metrics"]
    print(f"\n  Model:     {report['model']}")
    print(f"  Test file: {report['test_file']}")
    print(f"  Examples:  {report['num_examples']}")

    print(f"\n  {'Metric':<25} {'Value':>10} {'Target':>15}")
    print(f"  {'-'*50}")
    print(f"  {'Precision':<25} {agg['precision']:>10.4f} {'> 0.85':>15}")
    print(f"  {'Recall':<25} {agg['recall']:>10.4f} {'> 0.70':>15}")
    print(f"  {'F1 Score':<25} {agg['f1_score']:>10.4f} {'':>15}")
    print(f"  {'Severity Accuracy':<25} {report['severity_accuracy']:>10.4f} {'> 0.75':>15}")

    fpr = agg["total_false_positives"] / (
        agg["total_false_positives"] + agg["total_true_positives"]
    ) if (agg["total_false_positives"] + agg["total_true_positives"]) > 0 else 0.0
    print(f"  {'False Positive Rate':<25} {fpr:>10.4f} {'< 0.15':>15}")

    print(f"\n  Per-Category Breakdown:")
    print(f"  {'Category':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>5} {'FP':>5} {'FN':>5}")
    print(f"  {'-'*58}")
    for cat, metrics in report.get("per_category_metrics", {}).items():
        print(
            f"  {cat:<12} {metrics['precision']:>10.4f} "
            f"{metrics['recall']:>10.4f} {metrics['f1_score']:>10.4f} "
            f"{metrics['true_positives']:>5} {metrics['false_positives']:>5} "
            f"{metrics['false_negatives']:>5}"
        )

    print("\n" + "=" * 70)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the fine-tuned Go code review model"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the fine-tuned model directory",
    )
    parser.add_argument(
        "--test-file",
        default="dataset/processed/test.jsonl",
        help="Path to test JSONL file (default: dataset/processed/test.jsonl)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        help="Max number of examples to evaluate (for quick testing)",
    )
    parser.add_argument(
        "--output-report",
        help="Save evaluation report as JSON to this path",
    )

    args = parser.parse_args()

    report = evaluate(
        model_path=args.model,
        test_file=args.test_file,
        max_examples=args.max_examples,
    )

    print_report(report)

    if args.output_report:
        os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
        with open(args.output_report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n[INFO] Report saved to {args.output_report}")


if __name__ == "__main__":
    main()
