"""
api.py — FastAPI Serving Layer for Go Code Reviewer

Provides REST endpoints for code review, health checks, and report
generation.

Usage:
    uvicorn serving.api:app --host 0.0.0.0 --port 8000
    # or
    python -m serving.api

Endpoints:
    GET  /health           — Health check
    POST /review/code      — Review a code snippet
    POST /review/file      — Review an uploaded Go file
    POST /review/repo      — Review a local repository
    GET  /rules            — List all coding standards rules
"""

import os
import json
import time
import tempfile
import asyncio
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pipeline.review_pipeline import GoReviewPipeline, ReviewConfig


# ── Request / Response Models ────────────────────────────────────────────────

class CodeReviewRequest(BaseModel):
    """Request to review a Go code snippet."""
    code: str = Field(..., description="Go source code to review")
    file_path: str = Field(default="snippet.go", description="Virtual file path")
    mode: str = Field(default="hybrid", description="Review mode: hybrid | rag-only | fine-tune-only")
    top_k: int = Field(default=5, description="Number of rules to retrieve via RAG")


class RepoReviewRequest(BaseModel):
    """Request to review a local repository."""
    repo_path: str = Field(..., description="Absolute path to Go repository")
    mode: str = Field(default="hybrid", description="Review mode")
    output_format: str = Field(default="json", description="Output format: json | markdown | sarif")


class FindingResponse(BaseModel):
    """A single finding in the response."""
    rule_id: str = ""
    severity: str = ""
    category: str = ""
    title: str = ""
    file: str = ""
    line_start: int = 0
    line_end: int = 0
    function: str = ""
    description: str = ""
    current_code: str = ""
    suggested_fix: str = ""
    effort: str = ""
    auto_fixable: bool = False


class ReviewResponse(BaseModel):
    """Response from a code review request."""
    status: str = "success"
    findings: List[FindingResponse] = []
    total_findings: int = 0
    summary: dict = {}
    elapsed_seconds: float = 0.0
    raw_review: str = ""


class RepoReviewResponse(BaseModel):
    """Response from a repository review request."""
    status: str = "success"
    repo_path: str = ""
    total_files: int = 0
    total_chunks: int = 0
    total_findings: int = 0
    findings: List[FindingResponse] = []
    summary: dict = {}
    elapsed_seconds: float = 0.0
    report: Optional[str] = None  # Markdown/SARIF string if requested


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    model_loaded: bool = False
    rag_loaded: bool = False
    mode: str = ""
    uptime_seconds: float = 0.0


class RuleResponse(BaseModel):
    """A coding standard rule."""
    rule_id: str
    category: str
    severity: str
    title: str
    description: str


# ── Application State ────────────────────────────────────────────────────────

class AppState:
    """Shared application state."""
    def __init__(self):
        self.pipeline: Optional[GoReviewPipeline] = None
        self.start_time: float = time.time()
        self.review_count: int = 0
        self.rules: List[dict] = []


state = AppState()


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and resources on startup."""
    print("[API] Starting Go Code Reviewer API...")

    # Load config from environment
    config = ReviewConfig(
        model_path=os.getenv("MODEL_PATH", "./models/go-reviewer"),
        base_model=os.getenv("BASE_MODEL", "deepseek-ai/deepseek-coder-6.7b-instruct"),
        rag_db_path=os.getenv("RAG_DB_PATH", "./rag/qdrant_db"),
        mode=os.getenv("REVIEW_MODE", "hybrid"),
        ollama_model=os.getenv("OLLAMA_MODEL", None),
        ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        use_quantization=os.getenv("USE_QUANTIZATION", "true").lower() == "true",
    )

    state.pipeline = GoReviewPipeline(config=config)

    # Load pipeline in background to not block startup
    try:
        state.pipeline.load()
        print("[API] Pipeline loaded successfully.")
    except Exception as e:
        print(f"[API] Warning: Pipeline load failed: {e}")
        print("[API] API will start but reviews may fail.")

    # Load rules for /rules endpoint
    rules_path = config.rules_json
    if os.path.exists(rules_path):
        with open(rules_path, "r", encoding="utf-8") as f:
            state.rules = json.load(f)

    state.start_time = time.time()

    yield

    # Cleanup
    print("[API] Shutting down...")


# ── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Go Code Reviewer API",
    description="LLM-powered code review for Go repositories using RAG + Fine-tuning",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    pipeline = state.pipeline
    return HealthResponse(
        status="healthy" if pipeline else "degraded",
        model_loaded=pipeline.model is not None if pipeline else False,
        rag_loaded=pipeline.retriever is not None if pipeline else False,
        mode=pipeline.config.mode if pipeline else "",
        uptime_seconds=round(time.time() - state.start_time, 2),
    )


@app.post("/review/code", response_model=ReviewResponse)
async def review_code(request: CodeReviewRequest):
    """
    Review a Go code snippet.

    Accepts raw Go source code and returns structured review findings.
    """
    if not state.pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        # Override mode if requested
        original_mode = state.pipeline.config.mode
        if request.mode != state.pipeline.config.mode:
            state.pipeline.config.mode = request.mode

        start = time.time()
        result = state.pipeline.review_chunk(
            code=request.code,
            file_path=request.file_path,
            chunk_name=request.file_path,
        )

        # Restore mode
        state.pipeline.config.mode = original_mode
        state.review_count += 1

        from pipeline.deduplication import compute_summary
        summary = compute_summary(result.findings)

        return ReviewResponse(
            status="success",
            findings=[FindingResponse(**f) for f in result.findings],
            total_findings=len(result.findings),
            summary=summary,
            elapsed_seconds=round(time.time() - start, 2),
            raw_review=result.raw_review,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Review failed: {str(e)}")


@app.post("/review/file")
async def review_file(file: UploadFile = File(...)):
    """
    Review an uploaded Go file.

    Accepts a .go file upload and returns review findings.
    """
    if not state.pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if not file.filename.endswith(".go"):
        raise HTTPException(status_code=400, detail="Only .go files are accepted")

    try:
        content = await file.read()
        code = content.decode("utf-8")

        result = state.pipeline.review_chunk(
            code=code,
            file_path=file.filename,
            chunk_name=file.filename,
        )

        state.review_count += 1

        from pipeline.deduplication import compute_summary
        summary = compute_summary(result.findings)

        return ReviewResponse(
            status="success",
            findings=[FindingResponse(**f) for f in result.findings],
            total_findings=len(result.findings),
            summary=summary,
            elapsed_seconds=result.elapsed_seconds,
            raw_review=result.raw_review,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Review failed: {str(e)}")


@app.post("/review/repo", response_model=RepoReviewResponse)
async def review_repository(request: RepoReviewRequest):
    """
    Review a local Go repository.

    Scans the repository, chunks Go files, and reviews each chunk.
    This can take several minutes for large repositories.
    """
    if not state.pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if not os.path.isdir(request.repo_path):
        raise HTTPException(status_code=400, detail=f"Directory not found: {request.repo_path}")

    try:
        # Override mode
        original_mode = state.pipeline.config.mode
        state.pipeline.config.mode = request.mode

        report = state.pipeline.review_repository(request.repo_path)

        # Restore mode
        state.pipeline.config.mode = original_mode
        state.review_count += 1

        # Generate formatted report if requested
        formatted_report = None
        if request.output_format == "markdown":
            from pipeline.report_generator import generate_markdown_report
            formatted_report = generate_markdown_report(report)
        elif request.output_format == "sarif":
            from pipeline.report_generator import generate_sarif_report
            sarif = generate_sarif_report(report)
            formatted_report = json.dumps(sarif, indent=2)

        # Build response
        from dataclasses import asdict
        report_dict = asdict(report) if hasattr(report, "__dict__") else report

        return RepoReviewResponse(
            status="success",
            repo_path=report_dict.get("repo_path", ""),
            total_files=report_dict.get("total_files", 0),
            total_chunks=report_dict.get("total_chunks", 0),
            total_findings=report_dict.get("total_findings", 0),
            findings=[FindingResponse(**f) for f in report_dict.get("findings", [])],
            summary=report_dict.get("summary", {}),
            elapsed_seconds=report_dict.get("elapsed_seconds", 0.0),
            report=formatted_report,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Review failed: {str(e)}")


@app.get("/rules", response_model=List[RuleResponse])
async def list_rules():
    """List all coding standard rules."""
    return [
        RuleResponse(
            rule_id=r.get("rule_id", ""),
            category=r.get("category", ""),
            severity=r.get("severity", ""),
            title=r.get("title", ""),
            description=r.get("description", ""),
        )
        for r in state.rules
    ]


@app.get("/stats")
async def get_stats():
    """Get usage statistics."""
    return {
        "reviews_completed": state.review_count,
        "uptime_seconds": round(time.time() - state.start_time, 2),
        "mode": state.pipeline.config.mode if state.pipeline else "not loaded",
        "rules_count": len(state.rules),
    }


# ── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "serving.api:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("DEBUG", "false").lower() == "true",
        workers=1,  # Single worker for GPU models
    )
