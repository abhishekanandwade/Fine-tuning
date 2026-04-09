"""
review.py — CLI Entry Point for Go Code Reviewer

A rich command-line interface for reviewing Go repositories, files,
and code snippets using the RAG + Fine-tuned LLM pipeline.

Usage:
    # Review a repository
    python -m cli.review repo /path/to/go-repo --mode hybrid --format markdown

    # Review a single file
    python -m cli.review file /path/to/main.go

    # Review code from stdin
    echo 'package main...' | python -m cli.review snippet --stdin

    # Build RAG vector store
    python -m cli.review build-rag

    # Start API server
    python -m cli.review serve --port 8000
"""

import os
import sys

# Ensure the project root (code-review/) is on sys.path when running this
# script directly (e.g. `python cli/review.py …`), not just as a module.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import argparse
from typing import Optional


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="go-code-reviewer",
        description="LLM-powered Go code review using RAG + Fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
            %(prog)s repo ./my-go-project
            %(prog)s repo ./my-go-project --mode rag-only --format sarif -o report.sarif
            %(prog)s file ./main.go
            %(prog)s serve --port 8080
            %(prog)s build-rag --standards-dir ./standards
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── repo command ──
    repo_parser = subparsers.add_parser("repo", help="Review an entire Go repository")
    repo_parser.add_argument("path", help="Path to Go repository")
    repo_parser.add_argument("--model", default="./models/go-reviewer", help="Model path")
    repo_parser.add_argument("--base-model", default="deepseek-ai/deepseek-coder-6.7b-instruct")
    repo_parser.add_argument("--mode", choices=["hybrid", "rag-only", "fine-tune-only"], default="hybrid")
    repo_parser.add_argument("--format", "-f", choices=["json", "markdown", "sarif"], default="markdown")
    repo_parser.add_argument("--output", "-o", default=None, help="Output file (default: stdout)")
    repo_parser.add_argument("--rag-db", default="./rag/qdrant_db", help="Qdrant DB path")
    repo_parser.add_argument("--ollama", default=None, help="Use Ollama model (e.g., deepseek-coder)")
    repo_parser.add_argument("--no-quant", action="store_true", help="Disable quantization")
    repo_parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")

    # ── file command ──
    file_parser = subparsers.add_parser("file", help="Review a single Go file")
    file_parser.add_argument("path", help="Path to .go file")
    file_parser.add_argument("--model", default="./models/go-reviewer")
    file_parser.add_argument("--base-model", default="deepseek-ai/deepseek-coder-6.7b-instruct")
    file_parser.add_argument("--mode", choices=["hybrid", "rag-only", "fine-tune-only"], default="hybrid")
    file_parser.add_argument("--format", "-f", choices=["json", "markdown"], default="markdown")
    file_parser.add_argument("--output", "-o", default=None)
    file_parser.add_argument("--ollama", default=None)
    file_parser.add_argument("--no-quant", action="store_true")

    # ── snippet command ──
    snippet_parser = subparsers.add_parser("snippet", help="Review a Go code snippet")
    snippet_parser.add_argument("--stdin", action="store_true", help="Read code from stdin")
    snippet_parser.add_argument("--code", default=None, help="Go code string")
    snippet_parser.add_argument("--model", default="./models/go-reviewer")
    snippet_parser.add_argument("--base-model", default="deepseek-ai/deepseek-coder-6.7b-instruct")
    snippet_parser.add_argument("--mode", choices=["hybrid", "rag-only", "fine-tune-only"], default="hybrid")
    snippet_parser.add_argument("--ollama", default=None)
    snippet_parser.add_argument("--no-quant", action="store_true")

    # ── build-rag command ──
    rag_parser = subparsers.add_parser("build-rag", help="Build the RAG vector store")
    rag_parser.add_argument("--standards-dir", default="./standards")
    rag_parser.add_argument("--rules-json", default="./standards/rules.json")
    rag_parser.add_argument("--db-path", default="./rag/qdrant_db")
    rag_parser.add_argument("--rebuild", action="store_true")

    # ── serve command ──
    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--model", default="./models/go-reviewer")
    serve_parser.add_argument("--base-model", default="deepseek-ai/deepseek-coder-6.7b-instruct")
    serve_parser.add_argument("--mode", choices=["hybrid", "rag-only", "fine-tune-only"], default="hybrid")
    serve_parser.add_argument("--ollama", default=None)
    serve_parser.add_argument("--reload", action="store_true")

    # ── train command ──
    train_parser = subparsers.add_parser("train", help="Fine-tune the model")
    train_parser.add_argument("--config", default="./training/training_config.yaml")
    train_parser.add_argument("--model", default=None, help="Override base model")
    train_parser.add_argument("--epochs", type=int, default=None)
    train_parser.add_argument("--batch-size", type=int, default=None)

    return parser


def _make_config(args):
    """Build ReviewConfig from CLI args."""
    from pipeline.review_pipeline import ReviewConfig
    return ReviewConfig(
        model_path=getattr(args, "model", "./models/go-reviewer"),
        base_model=getattr(args, "base_model", "deepseek-ai/deepseek-coder-6.7b-instruct"),
        mode=getattr(args, "mode", "hybrid"),
        rag_db_path=getattr(args, "rag_db", "./rag/qdrant_db"),
        ollama_model=getattr(args, "ollama", None),
        use_quantization=not getattr(args, "no_quant", False),
    )


def cmd_repo(args):
    """Review an entire repository."""
    from pipeline.review_pipeline import GoReviewPipeline
    from pipeline.report_generator import (
        generate_markdown_report,
        generate_sarif_report,
        generate_json_report,
    )

    repo_path = os.path.abspath(args.path)
    if not os.path.isdir(repo_path):
        print(f"Error: Directory not found: {repo_path}", file=sys.stderr)
        sys.exit(1)

    config = _make_config(args)
    pipeline = GoReviewPipeline(config=config)
    report = pipeline.review_repository(repo_path)

    # Generate output
    if args.format == "markdown":
        output = generate_markdown_report(report)
    elif args.format == "sarif":
        sarif = generate_sarif_report(report)
        output = json.dumps(sarif, indent=2)
    else:
        output = generate_json_report(report)

    # Write output
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        if not getattr(args, "quiet", False):
            print(f"Report saved to {args.output}")
    else:
        print(output)


def cmd_file(args):
    """Review a single Go file."""
    from pipeline.review_pipeline import GoReviewPipeline
    from pipeline.report_generator import generate_markdown_report

    file_path = os.path.abspath(args.path)
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)
    if not file_path.endswith(".go"):
        print("Warning: File does not have .go extension", file=sys.stderr)

    config = _make_config(args)
    pipeline = GoReviewPipeline(config=config)
    findings = pipeline.review_file(file_path)

    if not findings:
        print("No violations found. Code looks good!")
        return

    # Simple output
    from pipeline.deduplication import compute_summary
    summary = compute_summary(findings)

    print(f"\n{'='*50}")
    print(f"  {len(findings)} issue(s) found in {os.path.basename(file_path)}")
    print(f"  Critical: {summary['critical']} | High: {summary['high']} "
          f"| Medium: {summary['medium']} | Low: {summary['low']}")
    print(f"{'='*50}\n")

    for finding in findings:
        severity = finding.get("severity", "LOW")
        icon = {"CRITICAL": "[!!]", "HIGH": "[!]", "MEDIUM": "[~]", "LOW": "[.]"}.get(severity, "[ ]")
        print(f"{icon} [{finding.get('rule_id', '')}] {severity} — {finding.get('title', '')}")
        print(f"    File: {finding.get('file', '')}:{finding.get('line_start', '')}")
        print(f"    {finding.get('description', '')[:120]}")
        if finding.get("suggested_fix"):
            print(f"    Fix available: Yes")
        print()


def cmd_snippet(args):
    """Review a code snippet."""
    from pipeline.review_pipeline import GoReviewPipeline

    if args.stdin:
        code = sys.stdin.read()
    elif args.code:
        code = args.code
    else:
        print("Error: Provide --stdin or --code", file=sys.stderr)
        sys.exit(1)

    if not code.strip():
        print("Error: Empty code input", file=sys.stderr)
        sys.exit(1)

    config = _make_config(args)
    pipeline = GoReviewPipeline(config=config)
    result = pipeline.review_chunk(code=code, file_path="snippet.go")

    if result.findings:
        for finding in result.findings:
            print(f"[{finding.get('rule_id', '')}] {finding.get('severity', '')} — {finding.get('title', '')}")
            print(f"  {finding.get('description', '')}")
            print()
    else:
        print("No violations found.")


def cmd_build_rag(args):
    """Build the RAG vector store."""
    from rag.build_vector_store import build_vector_store, verify_vector_store

    print("Building RAG vector store...")
    build_vector_store(
        standards_dir=args.standards_dir,
        rules_json=args.rules_json,
        db_path=args.db_path,
        rebuild=args.rebuild,
    )

    print("\nVerifying vector store...")
    verify_vector_store(db_path=args.db_path)
    print("Done!")


def cmd_serve(args):
    """Start the API server."""
    import uvicorn

    # Set env vars for the API
    os.environ["MODEL_PATH"] = getattr(args, "model", "./models/go-reviewer")
    os.environ["BASE_MODEL"] = getattr(args, "base_model", "deepseek-ai/deepseek-coder-6.7b-instruct")
    os.environ["REVIEW_MODE"] = getattr(args, "mode", "hybrid")
    if getattr(args, "ollama", None):
        os.environ["OLLAMA_MODEL"] = args.ollama

    print(f"Starting API server on {args.host}:{args.port}...")
    uvicorn.run(
        "serving.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=1,
    )


def cmd_train(args):
    """Run fine-tuning."""
    from training.fine_tune_go_reviewer import train

    # Build config overrides
    overrides = {}
    if args.model:
        overrides["model"] = args.model
    if args.epochs:
        overrides["epochs"] = args.epochs
    if args.batch_size:
        overrides["batch_size"] = args.batch_size

    train(config_path=args.config, **overrides)


def main():
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    commands = {
        "repo": cmd_repo,
        "file": cmd_file,
        "snippet": cmd_snippet,
        "build-rag": cmd_build_rag,
        "serve": cmd_serve,
        "train": cmd_train,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
