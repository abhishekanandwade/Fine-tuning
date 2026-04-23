# LLM-Based Code Review System for Go Repositories
## Industry-Standard Process: RAG + Fine-tuning Hybrid Approach

> **Goal:** Train and deploy a model on your custom Go coding standards so that when you provide any of your 30 repositories, the model automatically reviews all code and produces a structured, actionable report of required changes.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Phase 1 — Define & Encode Coding Standards](#phase-1--define--encode-coding-standards)
3. [Phase 2 — Choose Your Model Strategy](#phase-2--choose-your-model-strategy)
4. [Phase 3 — Build the Dataset Pipeline](#phase-3--build-the-dataset-pipeline)
5. [Phase 4 — Fine-tuning Pipeline](#phase-4--fine-tuning-pipeline)
6. [Phase 5 — RAG Layer Setup](#phase-5--rag-layer-setup)
7. [Phase 6 — Repository Review Pipeline](#phase-6--repository-review-pipeline)
8. [Phase 7 — Evaluation & Iteration](#phase-7--evaluation--iteration)
9. [Recommended Tech Stack](#recommended-tech-stack)
10. [Practical Rollout Timeline](#practical-rollout-timeline)
11. [Folder Structure](#folder-structure)

---

## Architecture Overview

The system follows a **Hybrid RAG + Fine-tuning** architecture, which is the industry standard for large organizations building internal code review automation tools.

```
┌─────────────────────────────────────────────────────────────────┐
│                    SYSTEM ARCHITECTURE                          │
│                                                                 │
│  ┌──────────────────┐        ┌──────────────────────────────┐  │
│  │  KNOWLEDGE LAYER │        │       INFERENCE LAYER        │  │
│  │                  │        │                              │  │
│  │  Standards Docs  │──────▶ │  Vector DB (Qdrant/Chroma)   │  │
│  │  (Markdown/JSON) │  embed │  Rule Retrieval (RAG)        │  │
│  └──────────────────┘        └──────────────┬───────────────┘  │
│                                             │ Top-K Rules       │
│  ┌──────────────────┐                       ▼                  │
│  │  TRAINING LAYER  │        ┌──────────────────────────────┐  │
│  │                  │        │     Fine-tuned LLM           │  │
│  │  Violation pairs │──────▶ │  (deepseek-coder / qwen2.5)  │  │
│  │  Real repo data  │ train  │  + LoRA / QLoRA adapters     │  │
│  └──────────────────┘        └──────────────┬───────────────┘  │
│                                             │                  │
│  ┌──────────────────┐                       ▼                  │
│  │   INPUT LAYER    │        ┌──────────────────────────────┐  │
│  │                  │        │      OUTPUT LAYER            │  │
│  │  Go Repository   │──────▶ │  Structured Review Report    │  │
│  │  AST Chunking    │ chunks │  (JSON + Markdown + SARIF)   │  │
│  └──────────────────┘        └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Why Hybrid RAG + Fine-tuning?**

| Approach | Pros | Cons |
|---|---|---|
| Prompting only | Fast to set up | Inconsistent, context window limits |
| Fine-tuning only | Consistent style | Doesn't retrieve fresh rules dynamically |
| RAG only | Always uses latest rules | No learned behavior for your patterns |
| **RAG + Fine-tuning** | **Best of both worlds** | **More setup, but production-grade** |

---

## Phase 1 — Define & Encode Coding Standards

### 1.1 Organize Standards as a Document Corpus

Create a structured folder of coding standards that will serve as both human documentation and the knowledge base for your RAG system.

```
standards/
  ├── naming_conventions.md       # Package, variable, function, constant naming
  ├── error_handling.md           # Error wrapping, sentinel errors, panic usage
  ├── concurrency_patterns.md     # Goroutine management, channel usage, mutex patterns
  ├── logging_standards.md        # Log levels, structured logging, context propagation
  ├── testing_standards.md        # Test structure, table-driven tests, mock usage
  ├── security_guidelines.md      # Input validation, secrets management, SQL injection
  ├── dependency_management.md    # go.mod standards, vendoring, version pinning
  ├── struct_design.md            # Interface design, embedding, constructor patterns
  ├── context_usage.md            # Context propagation, timeout handling, cancellation
  └── performance_guidelines.md   # Memory allocation, slice preallocation, escape analysis
```

### 1.2 Rule Format — Violation/Fix Pairs

For every standard you want enforced, document it as a machine-readable rule with before/after code examples. This format feeds directly into your fine-tuning dataset.

```json
{
  "rule_id": "EH-001",
  "category": "error_handling",
  "severity": "high",
  "title": "Always wrap errors with context",
  "description": "Bare error returns lose the call chain context. Use fmt.Errorf with %w to wrap errors so the full trace is preserved for debugging.",
  "violation_example": "func getUser(id int) (User, error) {\n    u, err := db.Query(id)\n    if err != nil {\n        return User{}, err  // BAD: loses context\n    }\n    return u, nil\n}",
  "correct_example": "func getUser(id int) (User, error) {\n    u, err := db.Query(id)\n    if err != nil {\n        return User{}, fmt.Errorf(\"getUser id=%d: %w\", id, err)  // GOOD\n    }\n    return u, nil\n}",
  "reference": "https://go.dev/blog/go1.13-errors",
  "auto_fixable": true
}
```

### 1.3 Categories of Standards to Cover

Document rules across these critical Go-specific categories:

| Category | Example Rules |
|---|---|
| **Error Handling** | Wrap with `%w`, no ignored errors, no `panic` in libraries |
| **Context Propagation** | First arg context, respect cancellation, set deadlines |
| **Naming** | Receiver names, acronym casing (URL not Url), unexported vs exported |
| **Concurrency** | WaitGroup usage, channel directionality, goroutine leak prevention |
| **Testing** | Table-driven tests, `t.Parallel()`, no sleep in tests, testify usage |
| **Logging** | Structured logging only, no `fmt.Println`, log at correct level |
| **Security** | No hardcoded secrets, parameterized queries, TLS configuration |
| **Performance** | Pre-allocate slices, avoid interface{} in hot paths, string builder usage |
| **Dependencies** | Minimal dependencies, go.sum committed, no indirect in main |
| **Documentation** | Exported functions documented, package-level doc comments |

---

## Phase 2 — Choose Your Model Strategy

### The Recommended Path: Hybrid RAG + Fine-tuning

```
Stage 1 (Week 1-2):  Start with GPT-4o + RAG to validate your standards
                     and generate initial training data cheaply.

Stage 2 (Week 4-6):  Fine-tune an open-source model on your curated dataset.
                     This removes API dependency and keeps code internal.

Stage 3 (Week 7+):   Hybrid deployment — fine-tuned model + RAG for freshness.
```

### Base Model Selection

Choose based on your hardware and privacy requirements:

| Model | Parameters | Hardware Needed | Best For | License |
|---|---|---|---|---|
| `deepseek-coder-33b-instruct` | 33B | 2x A100 or 4x RTX 4090 | Highest quality Go review | Open |
| `qwen2.5-coder-32b-instruct` | 32B | 2x A100 or 4x RTX 4090 | Multilingual, strong code | Open |
| `codellama-13b-instruct` | 13B | 1x RTX 4090 | Budget option, decent quality | Open |
| `deepseek-coder-7b-instruct` | 7B | 1x RTX 3090 | Fast iteration / prototyping | Open |
| `gpt-4o` (fine-tune API) | — | API only | If budget is not a constraint | Commercial |

> **Recommendation:** Start with `deepseek-coder-7b` for rapid prototyping. Once your dataset and evaluation pipeline are solid, move to `deepseek-coder-33b` or `qwen2.5-coder-32b` for production.

---

## Phase 3 — Build the Dataset Pipeline

### 3.1 Dataset Sources (in order of quality)

**Source 1: Manual Expert Examples (Highest Quality)**
Write 50–100 hand-crafted examples per category. Have your senior Go engineers review and validate each one. These are your "gold standard" examples.

**Source 2: Linter-Generated Labels (Scale)**
Run existing Go linters on your 30 repositories to automatically extract thousands of real violations from your own codebase:

```bash
# Install all linters
go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
go install honnef.co/go/tools/cmd/staticcheck@latest
go install github.com/securego/gosec/v2/cmd/gosec@latest
go install github.com/mgechev/revive@latest

# Run against a repo and output JSON
golangci-lint run ./... --out-format json > lint_output.json

# gosec for security rules
gosec -fmt json ./... > security_output.json
```

Then write a script to convert linter output into training pairs:
```python
# For each linter finding:
# 1. Extract the violating code snippet (AST-level)
# 2. Pair it with the linter's suggested fix or your standard's fix
# 3. Format as instruction-following training example
```

**Source 3: LLM-Assisted Synthesis (Augmentation)**
Use GPT-4o to generate additional synthetic violation/fix pairs for edge cases your linters don't cover. Always have a human review synthetic data.

### 3.2 Training Data Format

All training data must be in **JSONL (JSON Lines)** format, following the instruction-following chat template:

```jsonl
{"messages": [
  {
    "role": "system",
    "content": "You are an expert Go code reviewer enforcing company engineering standards. For each code snippet provided, identify all violations, explain why they violate standards, and provide the corrected code. Format your response with: VIOLATION [RULE-ID] SEVERITY, the issue description, the current code, and the suggested fix."
  },
  {
    "role": "user",
    "content": "Review the following Go function for standard violations:\n\n```go\nfunc processPayment(amount float64) error {\n    if amount <= 0 {\n        panic(\"invalid amount\")\n    }\n    result, err := paymentGateway.Charge(amount)\n    if err != nil {\n        return err\n    }\n    fmt.Println(\"Payment processed:\", result.ID)\n    return nil\n}\n```"
  },
  {
    "role": "assistant",
    "content": "## Code Review Findings\n\n### VIOLATION [EH-003] HIGH — Using panic in library/service code\n**File:** payment.go\n**Issue:** `panic` should never be used for recoverable error conditions in service code. It crashes the entire process and cannot be handled gracefully by callers.\n**Current code:**\n```go\npanic(\"invalid amount\")\n```\n**Suggested fix:**\n```go\nreturn fmt.Errorf(\"processPayment: amount must be positive, got %v\", amount)\n```\n\n### VIOLATION [EH-001] HIGH — Bare error return loses context\n**Issue:** Returning `err` directly loses the context of where and why the error occurred.\n**Current code:**\n```go\nreturn err\n```\n**Suggested fix:**\n```go\nreturn fmt.Errorf(\"processPayment: charging gateway: %w\", err)\n```\n\n### VIOLATION [LOG-002] MEDIUM — Using fmt.Println for logging\n**Issue:** `fmt.Println` is not structured logging and cannot be filtered, sampled, or enriched with context.\n**Current code:**\n```go\nfmt.Println(\"Payment processed:\", result.ID)\n```\n**Suggested fix:**\n```go\nlogger.Info(\"payment processed\", \"payment_id\", result.ID)\n```\n\n**Summary:** 3 violations found — 2 High, 1 Medium. Estimated fix effort: 15 minutes."
  }
]}
```

### 3.3 Dataset Size Targets

| Dataset Size | Expected Quality | Use Case |
|---|---|---|
| 200–500 examples | Good baseline | Proof of concept, 7B models |
| 500–1500 examples | Production ready | 13B–33B models |
| 1500–5000 examples | High precision | Large org, diverse Go patterns |

**Target:** Aim for **1000+ examples** covering all rule categories, with balanced representation across severity levels (not all high-severity examples).

### 3.4 Dataset Splits

```
dataset/
  ├── train.jsonl         # 80% of examples (~800 if 1000 total)
  ├── validation.jsonl    # 10% (~100) — used during training to prevent overfitting
  ├── test.jsonl          # 10% (~100) — held-out for final evaluation only
  └── rules_index/        # Markdown/JSON files for RAG vector store
```

---

## Phase 4 — Fine-tuning Pipeline

### 4.1 Technique: QLoRA (Quantized Low-Rank Adaptation)

**Why QLoRA?**
- Fine-tunes only ~1–2% of model parameters (the LoRA adapters)
- Runs a 33B model on 2x consumer GPUs via 4-bit quantization
- Training time: ~8–24 hours vs weeks for full fine-tuning
- Production proven at companies like Databricks, Hugging Face, Scale AI

```
Full Model Weights (frozen, 4-bit quantized)
         +
LoRA Adapter Weights (trainable, ~50MB)
         =
Fine-tuned Model Behavior
```

### 4.2 Training Script Structure

```python
# fine_tune_go_reviewer.py

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ── 1. Model & Quantization Config ──────────────────────────────────────────
MODEL_NAME = "deepseek-ai/deepseek-coder-7b-instruct-v1.5"  # start here

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4 — best quality
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,      # Nested quantization saves memory
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# ── 2. LoRA Configuration ────────────────────────────────────────────────────
lora_config = LoraConfig(
    r=64,                    # Rank — higher = more capacity for code patterns
    lora_alpha=128,          # Scaling factor (typically 2x rank)
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # Target all attention + feed-forward layers for best code quality
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── 3. Load Dataset ──────────────────────────────────────────────────────────
dataset = load_dataset("json", data_files={
    "train": "dataset/train.jsonl",
    "validation": "dataset/validation.jsonl",
})

# ── 4. Training Configuration ────────────────────────────────────────────────
sft_config = SFTConfig(
    output_dir="./go-reviewer-model",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,      # Effective batch size = 16
    gradient_checkpointing=True,        # Saves GPU memory
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    max_seq_length=8192,               # Go files can be long — keep high
    packing=False,                     # Keep examples separate for review tasks
    fp16=False,
    bf16=True,                         # Better than fp16 for modern GPUs
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True,
    report_to="wandb",                 # Track experiments
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=sft_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

trainer.train()
trainer.save_model("./go-reviewer-final")
```

### 4.3 Training Monitoring

Track these metrics during training using Weights & Biases (wandb):

| Metric | What to Watch For |
|---|---|
| `train/loss` | Should decrease steadily; plateau = training complete |
| `eval/loss` | Should track train loss; diverging = overfitting, reduce epochs |
| `train/learning_rate` | Cosine decay curve — confirms scheduler is working |
| GPU memory usage | Should stay below 90% — reduce batch size if OOM |

---

## Phase 5 — RAG Layer Setup

The RAG layer ensures your model always has access to the **latest and most relevant rules** when reviewing a file, even if those rules were added after training.

### 5.1 Embedding Your Standards

```python
# build_vector_store.py

from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import MarkdownTextSplitter
from langchain_community.document_loaders import DirectoryLoader
import json

# Load all standards documents
loader = DirectoryLoader("standards/", glob="**/*.md")
documents = loader.load()

# Also load rule JSON files
with open("standards/rules.json") as f:
    rules = json.load(f)

# Split into chunks (each rule = one chunk for precise retrieval)
splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# Use a code-aware embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1.5",  # Great for code + text
    model_kwargs={"trust_remote_code": True}
)

# Store in Qdrant (local mode for offline use)
vector_store = Qdrant.from_documents(
    chunks,
    embeddings,
    path="./qdrant_db",
    collection_name="go_coding_standards",
)

print(f"Indexed {len(chunks)} rule chunks into vector store")
```

### 5.2 Rule Retrieval at Inference Time

```python
# For each Go code chunk being reviewed:

def retrieve_relevant_rules(code_snippet: str, top_k: int = 5) -> str:
    """
    Given a Go code snippet, retrieve the most relevant coding standards.
    These rules are injected into the prompt alongside the code.
    """
    query = f"Go coding standards violations in: {code_snippet[:200]}"
    relevant_docs = vector_store.similarity_search(query, k=top_k)
    
    rules_text = "\n\n".join([
        f"**Rule {doc.metadata.get('rule_id', 'N/A')}:** {doc.page_content}"
        for doc in relevant_docs
    ])  
    return rules_text
```

---

## Phase 6 — Repository Review Pipeline

### 6.1 Go AST-Based Code Chunking

Parse Go source files using their Abstract Syntax Tree (AST) to extract meaningful, self-contained chunks rather than naive line-based splitting.

```python
# go_parser.py — uses subprocess to call Go's own toolchain

import subprocess
import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List

@dataclass
class GoChunk:
    file_path: str
    chunk_type: str          # "function", "method", "struct", "interface"
    name: str
    package: str
    start_line: int
    end_line: int
    code: str
    imports: List[str]

def extract_go_chunks(repo_path: str) -> List[GoChunk]:
    """
    Walk a Go repository and extract function/method/struct level chunks
    for individual review. This gives the LLM appropriately sized contexts.
    """
    chunks = []
    
    for go_file in Path(repo_path).rglob("*.go"):
        # Skip vendor and generated files
        if "vendor/" in str(go_file) or "_gen.go" in go_file.name:
            continue
        if "_test.go" in go_file.name:
            chunk_type_hint = "test"
        
        with open(go_file, "r", encoding="utf-8") as f:
            source = f.read()
        
        # Use 'go/ast' via a small Go helper script for accurate parsing
        file_chunks = parse_go_file(str(go_file), source)
        chunks.extend(file_chunks)
    
    return chunks

def chunk_strategy(file_lines: int) -> str:
    """Choose chunking granularity based on file size."""
    if file_lines < 150:
        return "whole_file"      # Review entire small files at once
    elif file_lines < 600:
        return "function_level"  # Review function by function
    else:
        return "method_level"    # Review individual methods for large files
```

### 6.2 Review Orchestration

```python
# review_pipeline.py

from transformers import pipeline
from typing import List, Dict
import json

class GoReviewPipeline:
    def __init__(self, model_path: str, vector_store):
        self.reviewer = pipeline(
            "text-generation",
            model=model_path,
            device_map="auto",
            torch_dtype="auto",
        )
        self.vector_store = vector_store
    
    def review_chunk(self, chunk: GoChunk) -> Dict:
        """Review a single Go code chunk."""
        
        # 1. Retrieve relevant rules from RAG
        relevant_rules = retrieve_relevant_rules(chunk.code)
        
        # 2. Build the review prompt
        prompt = f"""You are reviewing Go code against company engineering standards.

## Relevant Standards to Check Against:
{relevant_rules}

## Code to Review:
**File:** {chunk.file_path}
**Type:** {chunk.chunk_type} `{chunk.name}` in package `{chunk.package}`

```go
{chunk.code}
```

Review this code and identify all violations of the standards above.
For each violation provide: rule ID, severity, line number, description, current code, and suggested fix."""

        # 3. Run inference
        response = self.reviewer(
            prompt,
            max_new_tokens=1024,
            temperature=0.1,        # Low temperature for consistent reviews
            do_sample=True,
            repetition_penalty=1.1,
        )
        
        return {
            "file": chunk.file_path,
            "chunk_name": chunk.name,
            "chunk_type": chunk.chunk_type,
            "review": response[0]["generated_text"],
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
        }
    
    def review_repository(self, repo_path: str) -> Dict:
        """Full repository review — entry point."""
        
        print(f"[INFO] Extracting Go chunks from {repo_path}...")
        chunks = extract_go_chunks(repo_path)
        print(f"[INFO] Found {len(chunks)} reviewable chunks")
        
        all_findings = []
        
        for i, chunk in enumerate(chunks):
            print(f"[{i+1}/{len(chunks)}] Reviewing {chunk.chunk_type}: {chunk.name}")
            result = self.review_chunk(chunk)
            findings = parse_findings(result["review"])
            all_findings.extend(findings)
        
        # Deduplicate and rank by severity
        findings = deduplicate_findings(all_findings)
        findings = sorted(findings, key=lambda x: severity_rank(x["severity"]))
        
        return build_report(repo_path, findings)
```

### 6.3 Structured Output Report Format

The final report is generated in three formats for maximum usability:

**JSON (for tooling integration):**
```json
{
  "repository": "github.com/org/service-auth",
  "reviewed_at": "2026-04-06T10:30:00Z",
  "model_version": "go-reviewer-v1.2",
  "summary": {
    "total_files": 47,
    "total_chunks_reviewed": 312,
    "critical": 3,
    "high": 12,
    "medium": 28,
    "low": 15,
    "info": 8
  },
  "findings": [
    {
      "id": "finding-001",
      "file": "internal/auth/handler.go",
      "line_start": 47,
      "line_end": 52,
      "function": "ValidateToken",
      "rule_id": "SEC-003",
      "severity": "critical",
      "category": "security",
      "title": "JWT secret read from hardcoded string",
      "description": "Hardcoded secrets are a critical security vulnerability. They get committed to version control and are visible to anyone with repo access.",
      "current_code": "secret := \"mysupersecret123\"",
      "suggested_fix": "secret := os.Getenv(\"JWT_SECRET\")\nif secret == \"\" {\n    return nil, fmt.Errorf(\"JWT_SECRET env var not set\")\n}",
      "effort": "trivial",
      "auto_fixable": false
    }
  ]
}
```

**Markdown (for human review):**
```markdown
# Code Review Report: service-auth
Generated: 2026-04-06 | Model: go-reviewer-v1.2

## Summary
| Severity | Count |
|---|---|
| 🔴 Critical | 3 |
| 🟠 High | 12 |
| 🟡 Medium | 28 |
| 🟢 Low | 15 |

## Critical Findings

### [SEC-003] Hardcoded JWT Secret
**File:** `internal/auth/handler.go:47`
**Function:** `ValidateToken`
...
```

**SARIF (for GitHub/GitLab native integration):**
SARIF (Static Analysis Results Interchange Format) is the standard format used by GitHub Code Scanning. Outputting SARIF means your findings automatically appear in the GitHub Security tab of your repository.

---

## Phase 7 — Evaluation & Iteration

### 7.1 Evaluation Metrics

| Metric | Definition | Target |
|---|---|---|
| **Precision** | Of all flagged issues, what % are real violations? | > 85% |
| **Recall** | Of all real violations, what % did we catch? | > 70% |
| **False Positive Rate** | % of flagged items that are not violations | < 15% |
| **Human Agreement Rate** | % of findings a senior engineer agrees with | > 80% |
| **Severity Accuracy** | % of findings with correct severity assigned | > 75% |

### 7.2 Evaluation Process

```
Step 1: Run model on held-out test.jsonl (100 examples)
Step 2: Compare model output vs ground truth labels
        - Calculate precision, recall, F1 per category
Step 3: Sample 50 findings from real repo reviews
        - Have 2 senior Go engineers independently validate
        - Calculate inter-annotator agreement + human agreement rate
Step 4: Identify failure modes:
        - Which rule categories have lowest precision? (overfitting)
        - Which have lowest recall? (underfitting / data gaps)
Step 5: Add more training examples for weak categories
Step 6: Re-train and compare metrics — iterate
```

### 7.3 Red Flags to Watch For

- **Loss divergence during training** → Learning rate too high, reduce by 10x
- **Model copies code verbatim without finding issues** → Training data imbalance (too many "no violations" examples)
- **Model hallucinates violations** → Precision too low, add more negative examples (correct code), check rule definitions for ambiguity
- **Model misses severity** → Add explicit severity reasoning to training examples
- **Context window exceeded on large files** → Improve chunking strategy, add file-level summaries

---

## Recommended Tech Stack

| Component | Tool | Version | Purpose |
|---|---|---|---|
| **Base Model** | `deepseek-coder-33b-instruct` | latest | Core LLM for code review |
| **Fine-tuning** | Hugging Face `trl` (SFTTrainer) | ≥0.8 | Supervised fine-tuning |
| **PEFT/LoRA** | Hugging Face `peft` | ≥0.10 | Memory-efficient training |
| **Quantization** | `bitsandbytes` | ≥0.43 | 4-bit model loading |
| **Vector DB** | Qdrant (local mode) | ≥1.8 | Standards rule retrieval |
| **Embeddings** | `nomic-embed-text-v1.5` | latest | Code-aware text embeddings |
| **RAG Framework** | LangChain | ≥0.2 | RAG pipeline orchestration |
| **Go Parsing** | `go/ast` (stdlib) via subprocess | — | Accurate AST-level chunking |
| **Linting** | `golangci-lint` + `gosec` | latest | Training data generation |
| **Experiment Tracking** | Weights & Biases (`wandb`) | latest | Training monitoring |
| **Model Serving** | `vLLM` or `Ollama` | latest | Fast batched inference |
| **Output Format** | SARIF + JSON + Markdown | — | Multi-format reports |
| **CI Integration** | GitHub Actions | — | Per-PR automated review |

### Python Dependencies

```txt
# requirements.txt
torch>=2.2.0
transformers>=4.40.0
trl>=0.8.0
peft>=0.10.0
bitsandbytes>=0.43.0
datasets>=2.18.0
accelerate>=0.28.0
langchain>=0.2.0
langchain-community>=0.2.0
qdrant-client>=1.8.0
sentence-transformers>=2.7.0
wandb>=0.17.0
fastapi>=0.110.0        # For serving as an API
uvicorn>=0.29.0
python-dotenv>=1.0.0
```

---

## Practical Rollout Timeline

```
WEEK 1-2: FOUNDATION
├── Define all coding standards in markdown (target: 10 categories, 50+ rules)
├── Create 200 hand-crafted violation/fix example pairs (manual)
├── Set up Python environment + install dependencies
└── Run GPT-4o + raw standards prompt on 2-3 repos to validate standards quality

WEEK 3: DATA PIPELINE
├── Run golangci-lint + gosec on all 30 repositories
├── Write script to convert linter output → training JSONL format
├── Augment dataset to 800+ examples
└── Split into train/validation/test sets

WEEK 4: FIRST MODEL
├── Fine-tune deepseek-coder-7b with QLoRA (fast iteration, ~4 hours)
├── Evaluate on test set — calculate precision/recall per category
├── Identify top 3 failure categories → add more data for those
└── Build vector store from standards documents

WEEK 5-6: PRODUCTION MODEL
├── Fine-tune deepseek-coder-33b with QLoRA (~16-24 hours)
├── Integrate RAG layer with vector store
├── Build repository review CLI: python review.py --repo /path/to/repo
└── Compare 7B vs 33B results on test set

WEEK 7: REVIEW PIPELINE
├── Build Go AST chunking pipeline
├── Build report generation (JSON + Markdown + SARIF)
├── Run full review against all 30 repositories
└── Triage results with engineering team — collect feedback

WEEK 8: CI/CD INTEGRATION
├── Create GitHub Actions workflow for per-PR reviews
├── Set up SARIF upload to GitHub Security tab
├── Set up threshold alerts (fail PR if N+ critical findings)
└── Create Slack/email notification for nightly full-repo scans

WEEK 9+: CONTINUOUS IMPROVEMENT
├── Collect feedback from engineers on findings (thumbs up/down)
├── Use low-rated findings to identify model weaknesses
├── Quarterly re-training with new labeled data
└── A/B test new model versions before full rollout
```

---

## Folder Structure

```
go-code-reviewer/
├── standards/                         # Your coding standards corpus
│   ├── error_handling.md
│   ├── naming_conventions.md
│   ├── concurrency_patterns.md
│   ├── security_guidelines.md
│   └── rules.json                     # Machine-readable rule definitions
│
├── dataset/                           # Training data
│   ├── raw/                           # Linter outputs, manual examples
│   ├── processed/
│   │   ├── train.jsonl
│   │   ├── validation.jsonl
│   │   └── test.jsonl
│   └── build_dataset.py               # Script to generate dataset
│
├── training/                          # Fine-tuning code
│   ├── fine_tune_go_reviewer.py       # Main training script
│   ├── evaluate.py                    # Precision/recall evaluation
│   └── training_config.yaml          # Hyperparameters
│
├── rag/                               # RAG layer
│   ├── build_vector_store.py          # Index standards into Qdrant
│   ├── retriever.py                   # Rule retrieval functions
│   └── qdrant_db/                     # Vector store data (gitignored)
│
├── pipeline/                          # Review orchestration
│   ├── go_parser.py                   # AST-based Go code chunking
│   ├── review_pipeline.py             # Main orchestration
│   ├── report_generator.py            # JSON/Markdown/SARIF output
│   └── deduplication.py               # Finding dedup + ranking
│
├── serving/                           # Model serving
│   ├── api.py                         # FastAPI server
│   └── Dockerfile
│
├── cli/
│   └── review.py                      # CLI: python review.py --repo ./myrepo
│
├── .github/
│   └── workflows/
│       └── go-code-review.yml         # GitHub Actions CI workflow
│
├── models/                            # (gitignored — large files)
│   └── go-reviewer-final/
│
├── requirements.txt
├── README.md
└── go-code-review-plan.md             # This file
```

---

## Quick Start (After Reading This Plan)

```bash
# 1. Clone/set up project structure
mkdir go-code-reviewer && cd go-code-reviewer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Write your first 50 standards examples
# (Use standards/rules.json as your template)

# 4. Generate data from your repos
python dataset/build_dataset.py --repos-path /path/to/your/30/repos

# 5. Build vector store from standards
python rag/build_vector_store.py

# 6. Start with GPT-4o + RAG (no training needed yet)
python cli/review.py --repo ./sample-repo --mode rag-only

# 7. When satisfied with standards quality, fine-tune
python training/fine_tune_go_reviewer.py --config training/training_config.yaml

# 8. Run full pipeline
python cli/review.py --repo ./sample-repo --mode hybrid
```

---

*Last updated: April 6, 2026*
*Architecture: RAG + Fine-tuning Hybrid | Base Model: deepseek-coder-33b | Framework: Hugging Face TRL + QLoRA*
