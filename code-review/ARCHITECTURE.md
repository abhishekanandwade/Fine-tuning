# Go Code Review System — Architecture & Component Reference

This document explains every component of the `code-review/` system: what each module does, the classes and functions it exposes, how data flows between them, and how the pieces fit together at runtime.

---

## 1. High-Level Overview

The system is an **LLM-powered, RAG-augmented, multi-mode Go code reviewer**. It analyzes Go repositories for violations of internal coding standards (security, error handling, concurrency, naming, observability, performance, etc.) and emits structured findings (Markdown / JSON / SARIF).

### Operating Modes (`--mode`)

| Mode | Description |
|------|-------------|
| `hybrid` | RAG retrieval + fine-tuned model inference |
| `rag-only` | Retrieval-augmented with a base/Ollama model (no fine-tuning needed) |
| `fine-tune-only` | Fine-tuned model only, no retrieval |

### RAG Sub-modes (`--rag-mode`)

| Sub-mode | Description |
|----------|-------------|
| `simple` | Single-shot retrieval — fetch top-`k` rules per chunk and inject into prompt |
| `agentic` | LangGraph multi-agent — six specialist agents (security, architecture, performance, observability, database, concurrency) run in parallel with deterministic tool calls |

---

## 2. End-to-End Workflow

```
CLI / API request
      │
      ▼
GoReviewPipeline.load()
   ├── load model (local fine-tuned OR Ollama)
   ├── load RAG retriever (Qdrant + nomic embeddings)
   └── (agentic) load MultiAgentReviewer
      │
      ▼
extract_go_chunks(repo)         ← pipeline/go_parser.py (Go AST helper or regex)
      │
      ▼
For each GoChunk:
   ├── simple:   retriever.retrieve(code) → build prompt → _generate() → parse_findings()
   ├── agentic:  multi_agent.review_code(code) → planner → agents (parallel) → aggregator
   └── ft-only:  build prompt without rules → _generate()
      │
      ▼
deduplicate_findings() → rank_findings() → compute_summary()
      │
      ▼
RepositoryReport
      │
      ▼
generate_markdown_report() / generate_json_report() / generate_sarif_report()
```

---

## 3. Module-by-Module Reference

### 3.1 [pipeline/review_pipeline.py](pipeline/review_pipeline.py) — Main Orchestrator

CLI entry point and core orchestration class.

#### `ReviewConfig` (dataclass)
Pipeline configuration:
- `model_path`, `base_model` — local fine-tuned model / base model
- `mode` — `hybrid` | `rag-only` | `fine-tune-only`
- `rag_mode` — `simple` | `agentic`
- `ollama_model`, `ollama_url` — use remote Ollama instead of loading model locally
- `max_new_tokens` (2048), `temperature` (0 = deterministic), `top_p`, `seed`
- `top_k` (5) — rules retrieved per chunk
- `use_quantization` — 4-bit QLoRA loading
- `max_chunk_tokens` (3000) — input length cap
- `debug` — print raw LLM output

#### `ReviewFinding` (dataclass)
A single violation: `rule_id`, `severity`, `category`, `title`, `description`, `file`, `line_start`, `line_end`, `function`, `current_code`, `suggested_fix`, `effort`, `auto_fixable`.

#### `ReviewResult` (dataclass)
Output of one chunk: `chunk_name`, `chunk_file`, `raw_review`, `findings`, `elapsed_seconds`.

#### `RepositoryReport` (dataclass)
Final aggregate: `repo_path`, `total_files`, `total_chunks`, `total_findings`, `findings`, `summary`, `elapsed_seconds`, `config`.

#### `GoReviewPipeline` (main class)

| Method | Purpose |
|--------|---------|
| `load()` | Initialize model, tokenizer, RAG retriever, optional multi-agent orchestrator |
| `close()` | Free GPU memory, close Qdrant client |
| `review_repository(repo_path)` | Review entire repo → `RepositoryReport` |
| `review_chunk(code, file_path, chunk_name)` | Review single chunk → `ReviewResult` |
| `_load_local_model()` | Load fine-tuned model + LoRA adapter (or merged model) |
| `_load_ollama()` | Verify Ollama is reachable and model is pulled |
| `_load_retriever()` | Initialize `GoStandardsRetriever` and (if agentic) `MultiAgentReviewer` |
| `_build_prompt(code, rules_context)` | Compose user prompt |
| `_generate(system, user)` | Dispatch to local model or Ollama |
| `_load_static_rules()` | Fallback when RAG unavailable: load `standards/rules.json` directly |

**System prompt (abridged):**
```
You are an expert Go code reviewer. Find ALL violations.
For EVERY violation, output in this EXACT format:
### VIOLATION [RULE-ID] SEVERITY — Short Title
**File:** path/to/file.go:LINE
**Function:** functionName
**Issue:** Detailed description...
```

Notable details: BPE artifact cleanup for tokenizer mismatches; FlashAttention2 with eager/SDPA fallback; graceful degradation if RAG load fails.

---

### 3.2 [pipeline/go_parser.py](pipeline/go_parser.py) — Go AST Chunking

Splits a Go repo into reviewable units.

#### `GoChunk` (dataclass)
`file_path`, `chunk_type` (`function` | `method` | `struct` | `interface` | `whole_file`), `name`, `package`, `start_line`, `end_line`, `code`, `imports`, `receiver`, `doc_comment`.

#### Strategy
1. **Primary:** compile and run `ast_helper.go` (uses Go's `go/ast`) → emits JSON of every function/method/type with exact line ranges.
2. **Fallback:** regex matching if Go toolchain isn't available.

The helper binary is compiled into `tempfile.gettempdir()/go_code_reviewer/`. **Note:** if the system temp dir is shared (e.g. `/data/tmp`), set `TMPDIR` to a user-writable directory before running.

#### Public API
```python
extract_go_chunks(repo_path: str) -> List[GoChunk]
```
Skips `vendor/`, `.git/`, `node_modules/`.

---

### 3.3 [pipeline/deduplication.py](pipeline/deduplication.py) — Parsing, Dedup, Ranking

| Function | Purpose |
|----------|---------|
| `severity_rank(severity)` | Map name → numeric rank (CRITICAL=0 … INFO=4) |
| `parse_findings(review_text)` | Regex-extract structured findings from LLM output |
| `deduplicate_findings(findings)` | Drop near-duplicates (same rule+file+line) |
| `rank_findings(findings)` | Sort by severity, then line |
| `compute_summary(findings)` | `{"critical": n, "high": n, ...}` |
| `group_findings_by_file(findings)` | For per-file report sections |
| `group_findings_by_category(findings)` | For per-category report sections |

Primary regex matches `VIOLATION [RULE-ID] SEVERITY — Title` headers; a fallback regex tolerates the bracket-only form `[EH-001] HIGH ...`.

---

### 3.4 [pipeline/report_generator.py](pipeline/report_generator.py) — Output Formats

| Function | Format | Purpose |
|----------|--------|---------|
| `generate_markdown_report(report)` | Markdown | Human-readable report (overview, severity table, findings by file, findings by category) |
| `generate_json_report(report)` | JSON | Machine-readable with raw output and config |
| `generate_sarif_report(report)` | SARIF 2.1.0 | Standard format for GitHub / Azure DevOps integration |

---

### 3.5 [rag/retriever.py](rag/retriever.py) — Rule Retrieval

#### `GoStandardsRetriever`
```python
GoStandardsRetriever(
    db_path="rag/qdrant_db",
    collection_name="go_coding_standards",
    embedding_model="nomic-ai/nomic-embed-text-v1.5",  # 768-dim
)
```

| Method | Purpose |
|--------|---------|
| `retrieve(code, top_k=5, category_filter=None)` | Semantic search → `List[Document]` |
| `retrieve_with_scores(code, top_k=5, min_score=0.3)` | Same but returns similarity scores |
| `format_rules_for_prompt(rules)` | Render docs as plain text for prompts |
| `close()` | Cleanly close Qdrant client |

Includes a compatibility shim (`ensure_qdrant_search_compat`) that patches `search()` onto newer Qdrant clients that only expose `query_points()`.

---

### 3.6 [rag/build_vector_store.py](rag/build_vector_store.py) — Vector Store Builder

Indexes `standards/*.md` and `standards/rules.json` into Qdrant.

Pipeline:
1. Load markdown standards (DirectoryLoader).
2. Load `rules.json` — each rule becomes a Document with metadata.
3. Split with `MarkdownTextSplitter` (500 tokens, 50 overlap).
4. Embed with `nomic-embed-text-v1.5`.
5. Ingest into Qdrant collection `go_coding_standards` (cosine, 768-dim).
6. Persist to disk at `db_path`.

CLI:
```bash
python rag/build_vector_store.py
python rag/build_vector_store.py --rebuild
python rag/build_vector_store.py --standards-dir ./standards --db-path ./rag/qdrant_db
```

---

### 3.7 [rag/agent_tools.py](rag/agent_tools.py) — Deterministic Tool Functions

Plain Python functions used by the agentic mode. **Tools execute in Python, not via model tool-calling** — results are materialized into the prompt before the LLM is invoked.

| Tool | Purpose |
|------|---------|
| `tool_search_code(pattern, repo_path)` | Regex search across `.go` files |
| `tool_read_file(path, repo_path)` | Read file (≤200 KB) |
| `tool_list_directory(rel, repo_path)` | List dir contents |
| `tool_read_go_mod(repo_path)` | Read `go.mod` |
| `tool_parse_ast(path, repo_path)` | Lightweight AST summary (imports, funcs, types, goroutine/channel counts) |
| `tool_query_dependency_graph(repo_path)` | Build file → imports map |
| `tool_run_golangci_lint(repo_path)` | Run `golangci-lint` if installed |
| `tool_run_go_vet(repo_path)` | Run `go vet` if Go toolchain present |
| `tool_query_schema(repo_path)` | Collect DDL from `.sql` files |
| `tool_explain_query(sql)` | Static heuristics (SELECT \*, missing WHERE, N+1) |
| `tool_build_toolbox(repo_path, rag_retriever)` | Bundle all tools into a dict |

All tools degrade gracefully — missing binaries return `[tool-unavailable]`.

---

### 3.8 [rag/langgraph_agents.py](rag/langgraph_agents.py) — Multi-Agent Orchestration

Six specialist agents coordinated via a LangGraph `StateGraph`.

#### `AGENT_REGISTRY`

| Agent | Persona | Focus | RAG category | Rule IDs |
|-------|---------|-------|--------------|----------|
| `security` | AppSec Engineer | SQLi, auth, secrets, input validation | security | SEC-001/2/3 |
| `architecture` | Principal Architect | service boundaries, DI, contracts | — | NAM-001/2, DOC-001 |
| `performance` | Performance Engineer | N+1, allocations, leaks | performance | PERF-001 |
| `observability` | SRE / Platform | logging, traces, metrics, health | logging | LOG-001/2 |
| `database` | DB Reliability | tx safety, batches, indexes | security | SEC-002 |
| `concurrency` | Systems Engineer | goroutine lifecycle, channels, mutexes | concurrency | CONC-001/2, CTX-001/2 |

#### Graph
```
START → planner → [security | architecture | performance |
                   observability | database | concurrency]  (parallel)
                → aggregator → END
```

#### `ReviewState` (TypedDict)
`code`, `file_path`, `repo_path`, `selected_agents`, `findings` (append-only via `operator.add`), `agent_trace`.

#### `MultiAgentReviewer`
```python
MultiAgentReviewer(rag_retriever, generate_fn, repo_path, debug=False)
```

| Method | Purpose |
|--------|---------|
| `_build_graph()` | Construct the LangGraph |
| `_node_planner(state)` | LLM picks 2–5 relevant agents |
| `_make_agent_node(name)` | Factory: gather tool outputs → build prompt → single LLM call → parse findings |
| `_node_aggregator(state)` | Dedup + rank across agents |
| `review_code(code, file_path)` | Public API → `List[Dict]` findings |

Each agent makes **exactly one** LLM call per chunk; tool outputs are concatenated into the prompt up front.

---

### 3.9 [training/fine_tune_go_reviewer.py](training/fine_tune_go_reviewer.py) — QLoRA Fine-Tuning

QLoRA (4-bit NF4 + LoRA r=64, alpha=128) over `deepseek-coder-7b-instruct-v1.5` (or Qwen2.5-Coder).

Config knobs ([training/training_config.yaml](training/training_config.yaml)):
- 4-bit NF4 + double quantization, bf16 compute
- LoRA target modules: q/k/v/o_proj, gate/up/down_proj
- 3 epochs, batch=2, grad_accum=4, lr=2e-4 cosine, warmup 5 %
- max_seq_length=2048

Outputs: LoRA adapter at `output_dir/`, merged model at `final_model_dir/`.

Training data is JSONL chat format (`{"messages": [...]}`).

CLI:
```bash
python training/fine_tune_go_reviewer.py --config training/training_config.yaml
```

---

### 3.10 [training/evaluate_seeded.py](training/evaluate_seeded.py) — Seeded Benchmark Evaluation

Compares pipeline output (`results/seeded_review*.json`) against [benchmarks/ground_truth.json](benchmarks/ground_truth.json).

Per-file outcome: `PASS` / `MISS` / `NOISE` / `PARTIAL`.
Aggregate metrics: precision, recall, F1, false-positive rate.

```bash
python training/evaluate_seeded.py \
  --review ./results/seeded_review_agentic.json \
  --ground-truth ./benchmarks/ground_truth.json \
  --report ./results/eval_agentic.json
```

---

### 3.11 [training/evaluate.py](training/evaluate.py) — General Held-Out Evaluation

Runs the fine-tuned model over a JSONL test set, computing per-category and aggregate precision/recall/F1.

---

### 3.12 [dataset/build_dataset.py](dataset/build_dataset.py) — Dataset Construction

Converts linter outputs (golangci-lint, gosec) and manual examples into JSONL chat training data. Maps linter codes to internal rule IDs (e.g. `errcheck` → `EH-002`, gosec `G101` → `SEC-001`) and extracts ±15 lines of context per violation.

---

### 3.13 [cli/review.py](cli/review.py) — Rich CLI

Subcommands: `repo`, `file`, `snippet`, `build-rag`, `serve`.
Common flags: `--model`, `--base-model`, `--mode`, `--rag-db`, `--format`, `--output`, `--ollama`, `--no-quant`, `--quiet`.

---

### 3.14 [serving/api.py](serving/api.py) — FastAPI REST Server

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Liveness/readiness |
| `/review/code` | POST | Review a snippet |
| `/review/file` | POST | Review uploaded `.go` file |
| `/review/repo` | POST | Review a repo path on the server |
| `/rules` | GET | Dump all standards rules |

Pipeline is loaded once on app startup (lifespan context). Configuration via env vars: `MODEL_PATH`, `BASE_MODEL`, `RAG_DB_PATH`, `REVIEW_MODE`, `OLLAMA_MODEL`, `OLLAMA_URL`, `USE_QUANTIZATION`.

Run:
```bash
uvicorn serving.api:app --host 0.0.0.0 --port 8000
```

---

## 4. Standards & Rule Catalog

Source: [standards/](standards/)

### Security ([security_guidelines.md](standards/security_guidelines.md))
- **SEC-001** No hardcoded secrets
- **SEC-002** Parameterized SQL queries (no string concat)
- **SEC-003** Validate / sanitize external input
- **SEC-004** Use TLS for network I/O
- **SEC-005** Set timeouts on HTTP servers/clients

### Error Handling ([error_handling.md](standards/error_handling.md))
- **EH-001** Wrap errors with `fmt.Errorf("...: %w", err)`
- **EH-002** Never ignore errors
- **EH-003** No panic in library/service code
- **EH-004** Sentinel errors via `errors.Is`

### Concurrency ([concurrency_patterns.md](standards/concurrency_patterns.md))
- **CONC-001** Goroutines must have a termination path (ctx / done / WaitGroup)
- **CONC-002** Use directional channels (`chan<-`, `<-chan`)
- **CONC-003** Protect shared state with `sync.Mutex`
- **CONC-004** Use `sync.WaitGroup` for fan-out

### Naming ([naming_conventions.md](standards/naming_conventions.md))
- **NAM-001** MixedCaps, no underscores
- **NAM-002** Acronyms all-caps (`HTTPClient`, `userID`)
- **NAM-003** Short receiver names
- **NAM-004** Single-method interfaces end with `-er`
- **NAM-005** Lowercase single-word package names

### Context, Logging, Performance, Docs, Tests
- **CTX-001** `context.Context` is the first parameter
- **CTX-002** Never store context in a struct
- **LOG-001** No `fmt.Println` (use structured logger)
- **LOG-002** No `fmt.Printf` (use structured logger)
- **PERF-001** Pre-allocate slices when size is known
- **DOC-001** Exported funcs require doc comments
- **TEST-001** Tests should be table-driven

Structured form lives in [standards/rules.json](standards/rules.json) (each rule: `rule_id`, `category`, `severity`, `title`, `description`, `violation_example`, `correct_example`, `reference`, `auto_fixable`).

---

## 5. Benchmark Set

[benchmarks/seeded_repo/](benchmarks/seeded_repo/) — 18 Go files, each demonstrating a single seeded violation, plus one clean file. [ground_truth.json](benchmarks/ground_truth.json) maps each file to its expected `rule_id`/`severity`.

| File | Expected rule |
|------|---------------|
| `sec001_hardcoded.go` | SEC-001 / CRITICAL |
| `sec002_sqli.go` | SEC-002 / CRITICAL |
| `sec003_jwt_secret.go` | SEC-003 / CRITICAL |
| `eh001_bare_return.go` | EH-001 / HIGH |
| `eh002_ignored_error.go` | EH-002 / HIGH |
| `eh003_panic.go` | EH-003 / HIGH |
| `ctx001_ctx_not_first.go` | CTX-001 / HIGH |
| `ctx002_ctx_in_struct.go` | CTX-002 / MEDIUM |
| `log001_println.go` | LOG-001 / MEDIUM |
| `log002_printf.go` | LOG-002 / MEDIUM |
| `nam001_underscore.go` | NAM-001 / LOW |
| `nam002_acronym.go` | NAM-002 / LOW |
| `conc001_goroutine.go` | CONC-001 / HIGH |
| `conc002_bidirectional.go` | CONC-002 / MEDIUM |
| `test001_no_table.go` | TEST-001 / MEDIUM |
| `perf001_slice_alloc.go` | PERF-001 / MEDIUM |
| `doc001_no_comment.go` | DOC-001 / LOW |
| `clean_code.go` | (none — negative test) |

---

## 6. Data Formats

### Training example (JSONL line)
```json
{
  "messages": [
    {"role": "system",    "content": "You are an expert Go code reviewer..."},
    {"role": "user",      "content": "## Coding Standards...\n\n## Go Code\n```go\n...\n```"},
    {"role": "assistant", "content": "### VIOLATION [EH-001] HIGH — Bare return\n..."}
  ]
}
```

### Repository report (JSON)
```json
{
  "repo_path": "/path/to/repo",
  "total_files": 12,
  "total_chunks": 48,
  "total_findings": 23,
  "findings": [
    {
      "rule_id": "SEC-001",
      "severity": "CRITICAL",
      "category": "security",
      "title": "Hardcoded credentials",
      "file": "config.go",
      "line_start": 15,
      "line_end": 15,
      "function": "loadConfig",
      "description": "API key hardcoded in source",
      "current_code": "const apiKey = \"sk-abc123...\"",
      "suggested_fix": "apiKey := os.Getenv(\"API_KEY\")",
      "effort": "easy",
      "auto_fixable": true
    }
  ],
  "summary": {"critical": 1, "high": 5, "medium": 8, "low": 9, "info": 0},
  "elapsed_seconds": 45.3,
  "config": {"mode": "hybrid", "rag_mode": "simple", "top_k": 5}
}
```

---

## 7. Component Cheat Sheet

| Component | Purpose | Input | Output |
|-----------|---------|-------|--------|
| `pipeline/review_pipeline.py` | Orchestrator | repo / snippet | `RepositoryReport` |
| `pipeline/go_parser.py` | Go AST chunking | repo path | `List[GoChunk]` |
| `pipeline/deduplication.py` | Parse + dedup findings | LLM text | `List[dict]` |
| `pipeline/report_generator.py` | Format reports | `RepositoryReport` | Markdown / JSON / SARIF |
| `rag/retriever.py` | Vector retrieval | code snippet | `List[Document]` |
| `rag/build_vector_store.py` | Index standards | markdown + JSON | Qdrant DB |
| `rag/agent_tools.py` | Deterministic tools | repo path | dict of callables |
| `rag/langgraph_agents.py` | Multi-agent reviewer | code + repo path | `List[dict]` |
| `training/fine_tune_go_reviewer.py` | QLoRA fine-tune | JSONL data | LoRA adapter + merged model |
| `training/evaluate_seeded.py` | Seeded benchmark eval | review + ground truth | metrics report |
| `training/evaluate.py` | Held-out eval | model + test JSONL | per-category metrics |
| `dataset/build_dataset.py` | Build training data | linter output / repos | JSONL |
| `cli/review.py` | CLI front-end | CLI args | review output |
| `serving/api.py` | REST API | HTTP requests | JSON responses |

---

## 8. Common Operational Notes

- **Proxy / localhost:** the pipeline talks to Ollama at `http://localhost:11434`. If `HTTP_PROXY` / `HTTPS_PROXY` are set, ensure `no_proxy` includes both `localhost` and `127.0.0.1`.
- **Temp dir permissions:** `go_parser.py` compiles `ast_helper.go` in `tempfile.gettempdir()`. On shared hosts, `export TMPDIR=$HOME/.cache/go_reviewer_tmp` to avoid permission errors on `/data/tmp/go_code_reviewer/`.
- **HF cache:** the embedding model downloads under `HF_HOME` / `TRANSFORMERS_CACHE`. Permission-denied cache warnings are non-fatal but indicate a shared cache; point `HF_HOME` at a user-owned dir to silence them.
- **Determinism:** set `temperature=0` and a fixed `seed` in `ReviewConfig` for reproducible output.
