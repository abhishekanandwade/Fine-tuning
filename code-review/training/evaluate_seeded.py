"""
evaluate_seeded.py — Evaluate RAG pipeline output against the seeded benchmark.

Usage:
    # 1) Run the pipeline on the seeded repo:
    python pipeline/review_pipeline.py \
        --repo ./benchmarks/seeded_repo \
        --output ./results/seeded_review.json \
        --mode rag-only --ollama-model qwen2.5-coder:7b

    # 2) Score the output against ground truth:
    python training/evaluate_seeded.py \
        --review ./results/seeded_review.json \
        --ground-truth ./benchmarks/ground_truth.json \
        --report ./results/seeded_eval_report.json
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Set, Tuple


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_filename(path: str) -> str:
    """Keep just the basename for matching (pipeline emits absolute/relative paths)."""
    return os.path.basename((path or "").replace("\\", "/"))


def group_predictions_by_file(review: Dict) -> Dict[str, Set[str]]:
    """Return {filename: {rule_ids}} from a pipeline review JSON."""
    out: Dict[str, Set[str]] = defaultdict(set)
    for finding in review.get("findings", []):
        fname = normalize_filename(finding.get("file", ""))
        rid = (finding.get("rule_id") or "").upper()
        if fname and rid:
            out[fname].add(rid)
    return out


def group_expected_by_file(gt: Dict) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for fname, findings in (gt.get("files") or {}).items():
        out[fname] = {(f.get("rule_id") or "").upper() for f in findings if f.get("rule_id")}
    return out


def score(expected: Dict[str, Set[str]], predicted: Dict[str, Set[str]]) -> Dict:
    per_file = []
    total_tp = total_fp = total_fn = 0
    total_clean_correct = 0
    total_clean_files = 0

    for fname, exp_rules in expected.items():
        pred_rules = predicted.get(fname, set())
        tp = len(exp_rules & pred_rules)
        fp = len(pred_rules - exp_rules)
        fn = len(exp_rules - pred_rules)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Negative-test tracking (files with zero expected violations)
        if not exp_rules:
            total_clean_files += 1
            if not pred_rules:
                total_clean_correct += 1

        per_file.append({
            "file": fname,
            "expected": sorted(exp_rules),
            "predicted": sorted(pred_rules),
            "tp": tp, "fp": fp, "fn": fn,
            "status": (
                "PASS" if (exp_rules == pred_rules)
                else ("MISS" if fn and not fp else ("NOISE" if fp and not fn else "PARTIAL"))
            ),
        })

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = total_fp / (total_fp + total_tp) if (total_fp + total_tp) else 0.0
    neg_acc = total_clean_correct / total_clean_files if total_clean_files else 0.0

    # Per-rule breakdown
    per_rule = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    all_files = set(expected) | set(predicted)
    for fname in all_files:
        exp = expected.get(fname, set())
        pred = predicted.get(fname, set())
        for r in exp & pred:
            per_rule[r]["tp"] += 1
        for r in pred - exp:
            per_rule[r]["fp"] += 1
        for r in exp - pred:
            per_rule[r]["fn"] += 1

    rule_breakdown = {}
    for rid, s in sorted(per_rule.items()):
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        rule_breakdown[rid] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 4),
            "recall": round(r, 4),
        }

    return {
        "aggregate": {
            "files": len(expected),
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "false_positive_rate": round(fpr, 4),
            "clean_file_accuracy": round(neg_acc, 4),
        },
        "per_rule": rule_breakdown,
        "per_file": per_file,
    }


def print_report(report: Dict) -> None:
    agg = report["aggregate"]
    print("\n" + "=" * 68)
    print("         SEEDED BENCHMARK EVALUATION REPORT")
    print("=" * 68)
    print(f"  Files:                {agg['files']}")
    print(f"  True Positives (TP):  {agg['true_positives']}")
    print(f"  False Positives (FP): {agg['false_positives']}")
    print(f"  False Negatives (FN): {agg['false_negatives']}")
    print(f"  {'Metric':<28}{'Value':>10}{'Target':>16}")
    print(f"  {'-'*54}")
    print(f"  {'Precision':<28}{agg['precision']:>10.4f}{'> 0.85':>16}")
    print(f"  {'Recall':<28}{agg['recall']:>10.4f}{'> 0.70':>16}")
    print(f"  {'F1 Score':<28}{agg['f1_score']:>10.4f}{'':>16}")
    print(f"  {'False Positive Rate':<28}{agg['false_positive_rate']:>10.4f}{'< 0.15':>16}")
    print(f"  {'Clean-File Accuracy':<28}{agg['clean_file_accuracy']:>10.4f}{'1.0000':>16}")

    print("\n  Per-Rule Breakdown:")
    print(f"  {'Rule':<10}{'TP':>5}{'FP':>5}{'FN':>5}{'Prec':>10}{'Rec':>10}")
    print("  " + "-" * 45)
    for rid, m in report["per_rule"].items():
        print(f"  {rid:<10}{m['tp']:>5}{m['fp']:>5}{m['fn']:>5}{m['precision']:>10.4f}{m['recall']:>10.4f}")

    print("\n  Per-File Status:")
    print(f"  {'File':<30}{'Status':<10}{'Expected':<18}{'Predicted':<18}")
    print("  " + "-" * 74)
    for row in report["per_file"]:
        exp = ",".join(row["expected"]) or "-"
        pred = ",".join(row["predicted"]) or "-"
        print(f"  {row['file']:<30}{row['status']:<10}{exp:<18}{pred:<18}")
    print("=" * 68 + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--review", required=True, help="Pipeline review JSON output")
    p.add_argument("--ground-truth", default="benchmarks/ground_truth.json")
    p.add_argument("--report", default="results/seeded_eval_report.json")
    args = p.parse_args()

    review = load_json(args.review)
    gt = load_json(args.ground_truth)

    expected = group_expected_by_file(gt)
    predicted = group_predictions_by_file(review)

    report = score(expected, predicted)
    print_report(report)

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[INFO] Report saved to {args.report}")


if __name__ == "__main__":
    main()
