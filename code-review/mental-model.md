A normal repository review works like this:

CLI/API calls GoReviewPipeline
Pipeline loads:
model
tokenizer
retriever
Parser scans repo and extracts chunks
For each chunk:
retrieve top rules
build prompt
generate review text
parse findings from text
Merge all chunk findings
Deduplicate and rank
Generate Markdown / JSON / SARIF


standards/ = what “good Go code” means
dataset/ = convert code issues into training examples
training/ = teach the model those review patterns
rag/ = retrieve the most relevant rules at runtime
pipeline/ = run the actual review
cli/ = local developer usage
serving/ = HTTP service usage
.github/workflows/ = CI automation


Run Sequence

1) Install dependencies: from repo root run pip install -r requirements.txt using requirements.txt.
2) Prepare rules/standards: confirm rules.json and markdowns in standards are final.
3) Build training dataset (if training): run python [build_dataset.py](http://_vscodecontentref_/3) --repos-path <path-to-go-repos> --output-dir dataset/processed.
4) Fine-tune model: run python [fine_tune_go_reviewer.py](http://_vscodecontentref_/4) --config training/training_config.yaml using fine_tune_go_reviewer.py.
5) Evaluate model: run python [evaluate.py](http://_vscodecontentref_/6) --model ./go-reviewer-final --test-file dataset/processed/test.jsonl.
6) Build RAG vector DB: run python [build_vector_store.py](http://_vscodecontentref_/7) --standards-dir standards --rules-json [rules.json](http://_vscodecontentref_/8) --db-path rag/qdrant_db.
7) Run reviewer (CLI): run python -m cli.review repo <go-repo-path> --mode hybrid --format markdown -o report.md from review.py.
8) Or run as API: start python -m cli.review serve --port 8000, then call /review/code or /review/repo in api.py.
Practical shortcut (already-trained model)

Run only steps 1 → 2 → 6 → 7.
Important current caveat

Use python [build_vector_store.py](http://_vscodecontentref_/13) ... directly; python -m cli.review build-rag is likely mismatched with current function signature in build_vector_store.py.



Division of Responsibility (Rule of Thumb)

If it changes often or is policy text → RAG.
If it is stable response behavior/style → Fine-tune.
If uncertain: put rule text in RAG first; only fine-tune after repeated failures.