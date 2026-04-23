"""
retriever.py — Rule Retrieval Functions for RAG Layer

Provides functions to query the Qdrant vector store and retrieve
the most relevant coding standards given a Go code snippet.

Usage (as module):
    from rag.retriever import GoStandardsRetriever

    retriever = GoStandardsRetriever(db_path="rag/qdrant_db")
    rules = retriever.retrieve(code_snippet, top_k=5)
    formatted = retriever.format_rules_for_prompt(rules)
"""

import os
from types import MethodType
from typing import List, Dict, Optional

from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from qdrant_client import QdrantClient


# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
DEFAULT_COLLECTION_NAME = "go_coding_standards"
DEFAULT_TOP_K = 5


def ensure_qdrant_search_compat(client: QdrantClient) -> QdrantClient:
    """Patch QdrantClient instance to provide search() when only query_points() exists."""
    if hasattr(client, "search"):
        return client

    if not hasattr(client, "query_points"):
        return client

    def _search(self, *args, **kwargs):
        query_kwargs = dict(kwargs)
        if "query_vector" in query_kwargs:
            query_kwargs["query"] = query_kwargs.pop("query_vector")
        response = self.query_points(*args, **query_kwargs)
        return getattr(response, "points", response)

    client.search = MethodType(_search, client)
    return client


class GoStandardsRetriever:
    """
    Retrieves relevant Go coding standards from the vector store
    given a code snippet. Used at inference time in the review pipeline.
    """

    def __init__(
        self,
        db_path: str = "rag/qdrant_db",
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ):
        self.db_path = db_path
        self.collection_name = collection_name

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={
                "trust_remote_code": True,
                "device": "cpu",
            },
            encode_kwargs={"normalize_embeddings": True},
        )

        # Connect to existing Qdrant store
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"Vector store not found at {db_path}. "
                "Run 'python rag/build_vector_store.py' first."
            )

        self.client = ensure_qdrant_search_compat(QdrantClient(path=db_path))
        self.vector_store = Qdrant(
            client=self.client,
            collection_name=collection_name,
            embeddings=self.embeddings,
        )

    def close(self) -> None:
        """Explicitly close the Qdrant client to avoid shutdown-time ImportError."""
        try:
            if self.client is not None:
                self.client.close()
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()

    def retrieve(
        self,
        code_snippet: str,
        top_k: int = DEFAULT_TOP_K,
        category_filter: Optional[str] = None,
    ) -> List[Document]:
        """
        Retrieve the most relevant coding standards for a given code snippet.

        Args:
            code_snippet: Go source code to find relevant rules for
            top_k: Number of rules to retrieve
            category_filter: Optional category to filter by (e.g., "error_handling")

        Returns:
            List of Document objects with rule content and metadata
        """
        # Build query — include code context for better semantic matching
        query = f"Go coding standard violations in:\n{code_snippet[:500]}"

        if category_filter:
            # Use metadata filtering if supported
            results = self.vector_store.similarity_search(
                query,
                k=top_k,
                filter={"category": category_filter},
            )
        else:
            results = self.vector_store.similarity_search(query, k=top_k)

        return results

    def retrieve_with_scores(
        self,
        code_snippet: str,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = 0.3,
    ) -> List[tuple]:
        """
        Retrieve rules with similarity scores. Filters out low-relevance results.

        Returns:
            List of (Document, score) tuples
        """
        query = f"Go coding standard violations in:\n{code_snippet[:500]}"

        results = self.vector_store.similarity_search_with_score(query, k=top_k)

        # Filter by minimum score (higher score = more relevant)
        filtered = [(doc, score) for doc, score in results if score >= min_score]

        return filtered

    def format_rules_for_prompt(
        self,
        documents: List[Document],
        include_examples: bool = True,
    ) -> str:
        """
        Format retrieved rule documents into a text block suitable for
        injection into the LLM review prompt.

        Args:
            documents: Retrieved rule documents
            include_examples: Whether to include code examples in the output

        Returns:
            Formatted string of rules
        """
        if not documents:
            return "No specific standards matched. Apply general Go best practices."

        parts = []
        seen_rules = set()

        for doc in documents:
            rule_id = doc.metadata.get("rule_id", "N/A")

            # Deduplicate by rule_id
            if rule_id in seen_rules and rule_id != "N/A":
                continue
            seen_rules.add(rule_id)

            severity = doc.metadata.get("severity", "medium").upper()
            title = doc.metadata.get("title", "")
            category = doc.metadata.get("category", "general")

            if rule_id != "N/A":
                # Structured rule from JSON
                header = f"**[{rule_id}] {severity} — {title}**"
                parts.append(f"{header}\nCategory: {category}\n{doc.page_content}")
            else:
                # Chunk from markdown standards
                parts.append(doc.page_content)

        return "\n\n---\n\n".join(parts)

    def get_all_rule_ids(self) -> List[str]:
        """
        Get all unique rule IDs in the vector store.
        Useful for validation and debugging.
        """
        # Retrieve a large batch to get all rules
        results = self.vector_store.similarity_search(
            "Go coding standards rules", k=100
        )

        rule_ids = set()
        for doc in results:
            rid = doc.metadata.get("rule_id")
            if rid and rid != "N/A":
                rule_ids.add(rid)

        return sorted(rule_ids)

    def close(self):
        """Close the Qdrant client connection."""
        if hasattr(self, "client"):
            self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ── Standalone Retrieval Function ────────────────────────────────────────────

_global_retriever: Optional[GoStandardsRetriever] = None


def get_retriever(
    db_path: str = "rag/qdrant_db",
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> GoStandardsRetriever:
    """
    Get or create a singleton retriever instance.
    Avoid re-loading embeddings on every call.
    """
    global _global_retriever
    if _global_retriever is None:
        _global_retriever = GoStandardsRetriever(
            db_path=db_path,
            collection_name=collection_name,
        )
    return _global_retriever


def retrieve_relevant_rules(
    code_snippet: str,
    top_k: int = DEFAULT_TOP_K,
    db_path: str = "rag/qdrant_db",
) -> str:
    """
    Convenience function: retrieve and format relevant rules for a code snippet.
    Used directly by the review pipeline.

    Args:
        code_snippet: Go source code to find relevant rules for
        top_k: Number of rules to retrieve
        db_path: Path to the Qdrant database

    Returns:
        Formatted string of relevant rules ready for prompt injection
    """
    retriever = get_retriever(db_path=db_path)
    documents = retriever.retrieve(code_snippet, top_k=top_k)
    return retriever.format_rules_for_prompt(documents)


# ── CLI for Testing ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test rule retrieval")
    parser.add_argument(
        "--query",
        default='func getUser(id int) error { return err }',
        help="Go code snippet to search for relevant rules",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of rules to retrieve",
    )
    parser.add_argument(
        "--db-path",
        default="rag/qdrant_db",
        help="Path to Qdrant database",
    )

    args = parser.parse_args()

    print(f"Query: {args.query}")
    print(f"Top-K: {args.top_k}")
    print("=" * 60)

    result = retrieve_relevant_rules(args.query, args.top_k, args.db_path)
    print(result)
