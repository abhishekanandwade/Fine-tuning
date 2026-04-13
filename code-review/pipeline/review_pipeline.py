"""
review_pipeline.py — Core Review Orchestration

Ties together Go parsing, RAG retrieval, and LLM inference to produce
structured code review findings for an entire repository.

Usage (CLI):
    python -m pipeline.review_pipeline --repo /path/to/repo --model ./models/go-reviewer
    python -m pipeline.review_pipeline --repo /path/to/repo --mode rag-only

Usage (as module):
    from pipeline.review_pipeline import GoReviewPipeline
    pipeline = GoReviewPipeline(model_path="./models/go-reviewer")
    report = pipeline.review_repository("/path/to/repo")
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict

# Ensure the code-review root (parent of this file's directory) is on sys.path
# so that sibling packages like 'rag' and 'pipeline' are importable regardless
# of where the script is invoked from.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


# ── Configuration ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Go code reviewer. You know all Go best practices,
standard library patterns, and common pitfalls. Review the provided Go code against
the coding standards and rules supplied below.

For each violation found, output in this exact format:

### VIOLATION [RULE-ID] SEVERITY — Short Title
**File:** path/to/file.go:LINE
**Function:** functionName
**Issue:** Detailed description of the problem and why it matters.
**Current code:**
```go
// the violating code
```
**Suggested fix:**
```go
// the corrected code
```

If no violations are found, respond with: "No violations found."

Severity levels: CRITICAL, HIGH, MEDIUM, LOW, INFO

Be precise. Do not invent issues. Only flag real violations of the provided standards.
"""


@dataclass
class ReviewConfig:
    """Configuration for the review pipeline."""
    model_path: str = "./go-reviewer-final"
    base_model: str = "deepseek-ai/deepseek-coder-6.7b-instruct"
    rag_db_path: str = "./rag/qdrant_db"
    standards_dir: str = "./standards"
    rules_json: str = "./standards/rules.json"
    mode: str = "hybrid"  # "hybrid", "rag-only", "fine-tune-only"
    max_new_tokens: int = 2048
    temperature: float = 0.1
    top_p: float = 0.95
    top_k: int = 5  # Number of rules to retrieve per chunk
    batch_size: int = 1
    use_quantization: bool = True
    device: str = "auto"
    max_chunk_tokens: int = 3000
    ollama_model: Optional[str] = None  # Use Ollama instead of local model
    ollama_url: str = "http://localhost:11434"


@dataclass
class ReviewFinding:
    """A single review finding."""
    rule_id: str = ""
    severity: str = "LOW"
    category: str = ""
    title: str = ""
    file: str = ""
    line_start: int = 0
    line_end: int = 0
    function: str = ""
    description: str = ""
    current_code: str = ""
    suggested_fix: str = ""
    effort: str = "unknown"
    auto_fixable: bool = False
    confidence: float = 0.0


@dataclass
class ReviewResult:
    """Result of reviewing a single chunk."""
    chunk_name: str = ""
    chunk_file: str = ""
    raw_review: str = ""
    findings: List[Dict] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    tokens_used: int = 0


@dataclass
class RepositoryReport:
    """Complete report for a repository review."""
    repo_path: str = ""
    total_files: int = 0
    total_chunks: int = 0
    total_findings: int = 0
    findings: List[Dict] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    config: Dict = field(default_factory=dict)


# ── Pipeline Class ───────────────────────────────────────────────────────────

class GoReviewPipeline:
    """
    Main review pipeline that orchestrates Go parsing, RAG retrieval,
    and LLM-based code review.
    """

    def __init__(self, config: Optional[ReviewConfig] = None):
        self.config = config or ReviewConfig()
        self.model = None
        self.tokenizer = None
        self.retriever = None
        self._loaded = False

    def load(self):
        """Load all pipeline components (model, tokenizer, retriever)."""
        if self._loaded:
            return

        print("[INFO] Loading review pipeline components...")

        # ── Load RAG retriever ──
        if self.config.mode in ("hybrid", "rag-only"):
            self._load_retriever()

        # ── Load LLM ──
        if self.config.mode in ("hybrid", "fine-tune-only"):
            if self.config.ollama_model:
                self._load_ollama()
            else:
                self._load_local_model()

        self._loaded = True
        print("[INFO] Pipeline ready.")

    def close(self) -> None:
        """Release resources (model, retriever) explicitly."""
        if self.retriever is not None:
            try:
                self.retriever.close()
            except Exception:
                pass
            self.retriever = None
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *_):
        self.close()

    def _load_retriever(self):
        """Initialize the RAG retriever."""
        try:
            from rag.retriever import GoStandardsRetriever
            self.retriever = GoStandardsRetriever(
                db_path=self.config.rag_db_path,
                embedding_model="nomic-ai/nomic-embed-text-v1.5",
            )
            print(f"[INFO] RAG retriever loaded from {self.config.rag_db_path}")
        except Exception as e:
            print(f"[WARN] Failed to load RAG retriever: {e}")
            if self.config.mode == "rag-only":
                raise
            print("[WARN] Falling back to fine-tune-only mode.")
            self.config.mode = "fine-tune-only"

    def _load_local_model(self):
        """Load the fine-tuned model and tokenizer."""
        print(f"[INFO] Loading model from {self.config.model_path}...")

        # Quantization config
        quant_config = None
        if self.config.use_quantization:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Determine attention implementation safely
        attn_implementation = "eager"
        if torch.cuda.is_available():
            try:
                import flash_attn  # noqa: F401
                attn_implementation = "flash_attention_2"
                print("[INFO] FlashAttention2 available, using flash_attention_2.")
            except ImportError:
                print("[INFO] FlashAttention2 not installed, falling back to eager attention.")

        # Check if model_path is an adapter or full model
        adapter_config_path = os.path.join(self.config.model_path, "adapter_config.json")

        if os.path.exists(adapter_config_path):
            # Load base model + adapter
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                quantization_config=quant_config,
                device_map=self.config.device,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
            )
            self.model = PeftModel.from_pretrained(base_model, self.config.model_path)
            print("[INFO] Loaded LoRA adapter on base model.")
        else:
            # Load merged model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                quantization_config=quant_config,
                device_map=self.config.device,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
            )
            print("[INFO] Loaded merged model.")

        self.model.eval()
        print(f"[INFO] Model loaded on {self.model.device}")

    def _load_ollama(self):
        """Verify Ollama is available (actual calls done per-request).""" 
        import requests
        try:
            r = requests.get(f"{self.config.ollama_url}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            if self.config.ollama_model not in models:
                print(f"[WARN] Model '{self.config.ollama_model}' not found in Ollama. Available: {models}")
            else:
                print(f"[INFO] Ollama model '{self.config.ollama_model}' is available.")
        except requests.RequestException as e:
            print(f"[ERROR] Cannot connect to Ollama at {self.config.ollama_url}: {e}")
            raise

    # ── Inference ────────────────────────────────────────────────────────────

    def _build_prompt(self, code: str, rules_context: str) -> str:
        """Build the review prompt with code and relevant rules.""" 
        user_prompt = f"""## Applicable Coding Standards

{rules_context}

## Code to Review

```go
{code}
```

Review the above Go code against the provided coding standards. For each violation found, use the exact output format specified in the system prompt.
"""
        return user_prompt

    def _generate_with_local_model(self, system: str, user: str) -> str:
        """Generate review using local transformers model."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        try:
            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            input_text = f"### System:\n{system}\n\n### User:\n{user}\n\n### Assistant:\n"

        inputs = self.tokenizer(
            input_text, return_tensors="pt", truncation=True,
            max_length=self.config.max_chunk_tokens,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=1.1,
            )

        # Decode only the generated tokens
        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        return response.strip()

    def _generate_with_ollama(self, system: str, user: str) -> str:
        """Generate review using Ollama API."""
        import requests
        payload = {
            "model": self.config.ollama_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "num_predict": self.config.max_new_tokens,
            },
        }
        r = requests.post(
            f"{self.config.ollama_url}/api/chat",
            json=payload,
            timeout=300,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]

    def _generate(self, system: str, user: str) -> str:
        """Generate review with the configured backend."""
        if self.config.ollama_model:
            return self._generate_with_ollama(system, user)
        else:
            return self._generate_with_local_model(system, user)

    # ── Chunk Review ─────────────────────────────────────────────────────────

    def review_chunk(self, code: str, file_path: str = "", chunk_name: str = "") -> ReviewResult:
        """
        Review a single Go code chunk.

        Args:
            code: Go source code to review.
            file_path: Path of the file (for context).
            chunk_name: Name of the chunk (function/method).

        Returns:
            ReviewResult with raw review and parsed findings.
        """
        start = time.time()

        # ── Retrieve relevant rules via RAG ──
        rules_context = ""
        if self.retriever and self.config.mode in ("hybrid", "rag-only"):
            try:
                docs = self.retriever.retrieve(code, top_k=self.config.top_k)
                rules_context = self.retriever.format_rules_for_prompt(docs)
            except Exception as e:
                print(f"[WARN] RAG retrieval failed for {chunk_name}: {e}")

        if not rules_context:
            # Fallback: load rules.json directly
            rules_context = self._load_static_rules()

        # ── Build prompt and generate ──
        user_prompt = self._build_prompt(code, rules_context)
        print(f"  [INFO] Running inference for {chunk_name}...")

        if self.config.mode == "rag-only":
            # For rag-only mode, use a base model or Ollama without fine-tuning
            raw_review = self._generate(SYSTEM_PROMPT, user_prompt)
        else:
            raw_review = self._generate(SYSTEM_PROMPT, user_prompt)

        # ── Parse findings ──
        from pipeline.deduplication import parse_findings
        findings = parse_findings(raw_review)

        # Enrich findings with file path
        for f in findings:
            if not f.get("file"):
                f["file"] = file_path

        elapsed = time.time() - start

        return ReviewResult(
            chunk_name=chunk_name,
            chunk_file=file_path,
            raw_review=raw_review,
            findings=findings,
            elapsed_seconds=round(elapsed, 2),
        )

    def _load_static_rules(self) -> str:
        """Load rules from rules.json as fallback when RAG is unavailable."""
        rules_path = self.config.rules_json
        if not os.path.exists(rules_path):
            return "No coding standards available."

        with open(rules_path, "r", encoding="utf-8") as f:
            rules = json.load(f)

        lines = []
        for rule in rules:
            lines.append(f"### [{rule['rule_id']}] {rule['title']}")
            lines.append(f"Severity: {rule['severity']}")
            lines.append(f"Category: {rule['category']}")
            lines.append(rule["description"])
            if rule.get("violation_example"):
                lines.append(f"Bad: ```go\n{rule['violation_example']}\n```")
            if rule.get("correct_example"):
                lines.append(f"Good: ```go\n{rule['correct_example']}\n```")
            lines.append("")

        return "\n".join(lines)

    # ── Repository Review ────────────────────────────────────────────────────

    def review_repository(self, repo_path: str) -> RepositoryReport:
        """
        Review an entire Go repository.

        1. Parse all Go files into chunks.
        2. Review each chunk with RAG + LLM.
        3. Deduplicate and rank findings.
        4. Return consolidated report.
        """
        self.load()

        start = time.time()
        print(f"\n{'='*60}")
        print(f"  Reviewing: {repo_path}")
        print(f"  Mode: {self.config.mode}")
        print(f"{'='*60}\n")

        # ── Step 1: Parse Go files ──
        from pipeline.go_parser import extract_go_chunks
        chunks = extract_go_chunks(repo_path)
        print(f"[INFO] Parsed {len(chunks)} chunks from repository")

        if not chunks:
            print("[WARN] No Go code chunks found.")
            return RepositoryReport(repo_path=repo_path)

        # Count unique files
        unique_files = set(c.file_path for c in chunks)

        # ── Step 2: Review each chunk ──
        all_findings = []
        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks, 1):
            print(f"[{i}/{total_chunks}] Reviewing {chunk.name} in {chunk.file_path}...")

            result = self.review_chunk(
                code=chunk.code,
                file_path=chunk.file_path,
                chunk_name=chunk.name,
            )

            if result.findings:
                print(f"  → Found {len(result.findings)} issue(s) ({result.elapsed_seconds}s)")
                all_findings.extend(result.findings)
            else:
                print(f"  → Clean ({result.elapsed_seconds}s)")

        # ── Step 3: Deduplicate and rank ──
        from pipeline.deduplication import deduplicate_findings, rank_findings, compute_summary
        deduped_findings = deduplicate_findings(all_findings)
        ranked_findings = rank_findings(deduped_findings)

        # ── Step 4: Build report ──
        summary = compute_summary(ranked_findings)
        elapsed = time.time() - start

        report = RepositoryReport(
            repo_path=repo_path,
            total_files=len(unique_files),
            total_chunks=total_chunks,
            total_findings=len(ranked_findings),
            findings=ranked_findings,
            summary=summary,
            elapsed_seconds=round(elapsed, 2),
            config=asdict(self.config),
        )

        print(f"\n{'='*60}")
        print(f"  Review Complete")
        print(f"  Files: {report.total_files} | Chunks: {report.total_chunks}")
        print(f"  Findings: {report.total_findings} "
              f"(Critical: {summary.get('critical', 0)}, "
              f"High: {summary.get('high', 0)}, "
              f"Medium: {summary.get('medium', 0)}, "
              f"Low: {summary.get('low', 0)})")
        print(f"  Time: {report.elapsed_seconds}s")
        print(f"{'='*60}\n")

        return report

    def review_file(self, file_path: str) -> List[Dict]:
        """Review a single Go file and return findings."""
        self.load()

        from pipeline.go_parser import parse_go_file_regex, chunk_strategy, GoChunk

        if not os.path.exists(file_path):
            print(f"[ERROR] File not found: {file_path}")
            return []

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        strategy = chunk_strategy(content)

        if strategy == "whole_file":
            chunks = [GoChunk(
                file_path=file_path,
                chunk_type="file",
                name=os.path.basename(file_path),
                code=content,
                start_line=1,
                end_line=len(content.split("\n")),
            )]
        else:
            chunks = parse_go_file_regex(content, file_path)

        all_findings = []
        for chunk in chunks:
            result = self.review_chunk(
                code=chunk.code,
                file_path=file_path,
                chunk_name=chunk.name,
            )
            all_findings.extend(result.findings)

        from pipeline.deduplication import deduplicate_findings, rank_findings
        return rank_findings(deduplicate_findings(all_findings))


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Go Code Review Pipeline")
    parser.add_argument("--repo", required=True, help="Path to Go repository")
    parser.add_argument("--model", default="./go-reviewer-final", help="Path to fine-tuned model")
    parser.add_argument("--base-model", default="deepseek-ai/deepseek-coder-6.7b-instruct")
    parser.add_argument("--mode", choices=["hybrid", "rag-only", "fine-tune-only"], default="hybrid")
    parser.add_argument("--rag-db", default="./rag/qdrant_db", help="Path to Qdrant DB")
    parser.add_argument("--ollama-model", default=None, help="Ollama model name (e.g., deepseek-coder)")
    parser.add_argument("--output", default=None, help="Output report JSON path")
    parser.add_argument("--format", choices=["json", "markdown", "sarif"], default="json")
    parser.add_argument("--no-quant", action="store_true", help="Disable quantization")

    args = parser.parse_args()

    config = ReviewConfig(
        model_path=args.model,
        base_model=args.base_model,
        mode=args.mode,
        rag_db_path=args.rag_db,
        use_quantization=not args.no_quant,
        ollama_model=args.ollama_model,
    )

    with GoReviewPipeline(config=config) as pipeline:
        report = pipeline.review_repository(args.repo)

    # Generate output
    if args.format == "json":
        output_data = asdict(report)
    elif args.format == "markdown":
        from pipeline.report_generator import generate_markdown_report
        output_data = generate_markdown_report(report)
    elif args.format == "sarif":
        from pipeline.report_generator import generate_sarif_report
        output_data = generate_sarif_report(report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            if isinstance(output_data, str):
                f.write(output_data)
            else:
                json.dump(output_data, f, indent=2, default=str)
        print(f"[INFO] Report saved to {args.output}")
    else:
        if isinstance(output_data, str):
            print(output_data)
        else:
            print(json.dumps(output_data, indent=2, default=str))


if __name__ == "__main__":
    main()
