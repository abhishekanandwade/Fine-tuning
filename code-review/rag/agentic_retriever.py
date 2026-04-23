"""
agentic_retriever.py — Agentic RAG layer for code review

Difference vs. simple RAG (`retriever.py`):
    Simple RAG: 1 embedding query → top-K rules → 1 LLM call.
    Agentic RAG:
        Step 1 PLAN     — LLM inspects the code and picks the most
                          relevant rule *categories* to investigate.
        Step 2 RETRIEVE — Issue one targeted query per chosen category
                          (with metadata filter) and merge / dedupe.
        Step 3 REVIEW   — LLM reviews the code with the focused context.
        Step 4 VERIFY   — For every finding, LLM is asked "is this truly
                          a violation? (yes/no + reason)"; non-yes
                          findings are dropped.

The agent is pragmatic: it uses small targeted LLM calls instead of a
heavy LangChain agent loop, so it works with the existing Ollama backend
and stays deterministic with temperature=0.
"""

from __future__ import annotations

import json
import re
from typing import Callable, Dict, List, Optional

from langchain_core.documents import Document

from rag.retriever import GoStandardsRetriever


# Categories must match the `category` metadata stored in the vector store
# (see standards/rules.json). Keep this list in sync with rules.json.
KNOWN_CATEGORIES = [
    "security",
    "error_handling",
    "context_usage",
    "logging",
    "naming",
    "concurrency",
    "testing",
    "performance",
    "documentation",
]


# ── Prompts used by the agent ────────────────────────────────────────────────

PLANNER_SYSTEM = """You are a Go code review planner.
Given a Go code snippet, decide which rule CATEGORIES are worth checking.

Output STRICT JSON only — no prose, no code fences:
{"categories": ["category1", "category2", ...]}

Pick 2 to 4 categories from this fixed list:
security, error_handling, context_usage, logging, naming,
concurrency, testing, performance, documentation

Choose categories that are LIKELY to apply based on what the code does.
If unsure, include the most obviously relevant ones."""


VERIFIER_SYSTEM = """You are a strict Go code review verifier.
You will be given (a) a single proposed violation and (b) the original code.

Decide whether the proposed violation is a TRUE positive in this code.
Output STRICT JSON only — no prose, no code fences:
{"verdict": "yes" | "no", "reason": "<one sentence>"}

Be strict: if the violating pattern is not actually present in the code,
answer "no"."""


# ── Agentic Retriever ────────────────────────────────────────────────────────


class AgenticGoStandardsRetriever:
    """
    Multi-step RAG retriever that uses small LLM calls to plan retrieval
    and to verify findings. Wraps the simple `GoStandardsRetriever` for
    the actual vector search work.
    """

    def __init__(
        self,
        base_retriever: GoStandardsRetriever,
        generate_fn: Callable[[str, str], str],
        per_category_top_k: int = 3,
        max_categories: int = 4,
        verify_findings: bool = True,
        debug: bool = False,
    ):
        """
        Args:
            base_retriever: The underlying simple retriever (vector store wrapper).
            generate_fn:    Callable(system_prompt, user_prompt) -> str.
                            Use the pipeline's existing LLM backend (Ollama or local).
            per_category_top_k: How many rules to fetch per chosen category.
            max_categories: Hard cap on how many categories the planner can pick.
            verify_findings: If True, run the verification step to drop FPs.
            debug: Print intermediate planner / verifier output.
        """
        self.base = base_retriever
        self.generate = generate_fn
        self.per_category_top_k = per_category_top_k
        self.max_categories = max_categories
        self.verify_findings = verify_findings
        self.debug = debug

    # Pipeline expects retrievers to expose `close()`
    def close(self) -> None:
        try:
            self.base.close()
        except Exception:
            pass

    # ── Step 1: PLAN ────────────────────────────────────────────────────────

    def _plan_categories(self, code_snippet: str) -> List[str]:
        """Ask the LLM which rule categories are worth investigating."""
        user = f"```go\n{code_snippet[:1500]}\n```"
        try:
            raw = self.generate(PLANNER_SYSTEM, user)
        except Exception as e:
            if self.debug:
                print(f"[AGENTIC] Planner call failed: {e} — falling back to all categories")
            return KNOWN_CATEGORIES[: self.max_categories]

        cats = _extract_json_list(raw, key="categories")

        # Sanitize: keep only known categories, dedupe, cap length
        clean: List[str] = []
        for c in cats:
            c_norm = str(c).strip().lower()
            if c_norm in KNOWN_CATEGORIES and c_norm not in clean:
                clean.append(c_norm)
            if len(clean) >= self.max_categories:
                break

        if not clean:
            # Planner returned nothing usable — fall back to a safe default
            clean = ["security", "error_handling", "context_usage"]

        if self.debug:
            print(f"[AGENTIC] Planner chose categories: {clean}")
        return clean

    # ── Step 2: RETRIEVE ────────────────────────────────────────────────────

    def retrieve(self, code_snippet: str, top_k: int = 5) -> List[Document]:
        """
        Plan + multi-query retrieval. The `top_k` argument is kept for
        interface compatibility but ignored: agentic uses
        per_category_top_k * num_categories instead.
        """
        categories = self._plan_categories(code_snippet)

        merged: List[Document] = []
        seen_rule_ids = set()

        for cat in categories:
            try:
                docs = self.base.retrieve(
                    code_snippet,
                    top_k=self.per_category_top_k,
                    category_filter=cat,
                )
            except Exception as e:
                if self.debug:
                    print(f"[AGENTIC] Filtered retrieval failed for {cat}: {e}")
                docs = []

            # Some vector backends don't honor metadata filters reliably;
            # if we got nothing, fall back to a plain query.
            if not docs:
                docs = self.base.retrieve(
                    f"{cat} rule violations in:\n{code_snippet[:500]}",
                    top_k=self.per_category_top_k,
                )

            for d in docs:
                rid = d.metadata.get("rule_id", id(d))
                if rid in seen_rule_ids:
                    continue
                seen_rule_ids.add(rid)
                merged.append(d)

        if self.debug:
            ids = [d.metadata.get("rule_id", "?") for d in merged]
            print(f"[AGENTIC] Retrieved {len(merged)} rules across categories: {ids}")
        return merged

    # ── Helper kept identical to simple retriever ───────────────────────────

    def format_rules_for_prompt(
        self,
        documents: List[Document],
        include_examples: bool = True,
    ) -> str:
        return self.base.format_rules_for_prompt(documents, include_examples)

    # ── Step 4: VERIFY ──────────────────────────────────────────────────────

    def verify_findings_against_code(
        self,
        findings: List[Dict],
        code_snippet: str,
    ) -> List[Dict]:
        """
        Self-critique pass. Drops findings the verifier rejects.
        Safe to call with any finding-dict shape — it only reads a few fields.
        """
        if not self.verify_findings or not findings:
            return findings

        kept: List[Dict] = []
        for f in findings:
            user = (
                "Proposed violation:\n"
                f"  rule_id: {f.get('rule_id', '')}\n"
                f"  severity: {f.get('severity', '')}\n"
                f"  description: {f.get('description', '')[:300]}\n"
                f"  current_code: {f.get('current_code', '')[:300]}\n\n"
                "Original code under review:\n"
                f"```go\n{code_snippet[:1500]}\n```"
            )
            try:
                raw = self.generate(VERIFIER_SYSTEM, user)
            except Exception as e:
                if self.debug:
                    print(f"[AGENTIC] Verifier call failed for {f.get('rule_id')}: {e}")
                kept.append(f)  # fail-open: keep finding if verifier errors
                continue

            verdict = _extract_json_value(raw, "verdict", default="yes").lower()
            if self.debug:
                reason = _extract_json_value(raw, "reason", default="")
                print(f"[AGENTIC] Verify {f.get('rule_id')}: {verdict} — {reason}")
            if verdict.startswith("y"):
                kept.append(f)
        return kept


# ── JSON extraction helpers (LLM output is sometimes messy) ─────────────────


def _extract_json_block(text: str) -> Optional[dict]:
    """Find the first {...} JSON object in `text` and parse it."""
    if not text:
        return None
    # Strip code fences if present
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "")
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _extract_json_list(text: str, key: str) -> List[str]:
    obj = _extract_json_block(text) or {}
    val = obj.get(key, [])
    if isinstance(val, list):
        return [str(x) for x in val]
    return []


def _extract_json_value(text: str, key: str, default: str = "") -> str:
    obj = _extract_json_block(text) or {}
    val = obj.get(key, default)
    return str(val) if val is not None else default
