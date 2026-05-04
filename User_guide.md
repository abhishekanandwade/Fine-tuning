cd "C:\Users\knchaitr\OneDrive - Hewlett Packard Enterprise\CoE Team\Fine-tuning\code-review"
$env:PYTHONIOENCODING="utf-8"

# 1. Initial agentic-RAG review (LLM-confirmed). Takes ~5–15 min on this repo.
python -m pipeline.architectural_rag_agent `
  --repo "..\post-data-management-back-end-development" `
  --output results\architectural_review.json `
  --ollama-model qwen2.5-coder:7b

# 2. Grade coverage vs deterministic baseline (lists missed violations).
python -m pipeline.evaluate_coverage `
  --repo "..\post-data-management-back-end-development" `
  --report results\architectural_review.json `
  --output results\coverage_report.json `
  --markdown results\coverage_report.md

# 3. Recover the missed ones (instant, deterministic).
python -m pipeline.architectural_rag_agent `
  --repo "..\post-data-management-back-end-development" `
  --output results\architectural_review_recovered.json `
  --recover-from results\coverage_report.json --no-llm

# 4. Merge into final report + final coverage.
python -m pipeline.merge_findings `
  --inputs results\architectural_review.json results\architectural_review_recovered.json `
  --output results\architectural_review_merged.json

python -m pipeline.evaluate_coverage `
  --repo "..\post-data-management-back-end-development" `
  --report results\architectural_review_merged.json `
  --output results\coverage_report_merged.json `
  --markdown results\coverage_report_merged.md

# Final outputs to read

1. Findings: results/         architectural_review_merged.json — 75 violations across REPO-001, REPO-002, HANDLER-001
2. Grade: results/coverage_report_merged.md — should show 100% recall

# Prerequisites (one-time)

# venv already exists; activate it
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& "..\.venv\Scripts\Activate.ps1"

# Ollama must be running with the model pulled
ollama pull qwen2.5-coder:7b
ollama serve   # or run as a service

# One-shot fast mode (skip the LLM entirely)
1. If you trust the deterministic pre-filter and want results in under a second:

python -m pipeline.architectural_rag_agent `
  --repo "..\post-data-management-back-end-development" `
  --output results\architectural_review_merged.json --no-llm

  