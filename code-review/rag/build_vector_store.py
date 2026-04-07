"""
build_vector_store.py — Index Go Coding Standards into Qdrant Vector DB

Embeds all coding standard documents (markdown + JSON rules) into a
Qdrant vector database for retrieval at inference time (RAG layer).

Usage:
    python rag/build_vector_store.py
    python rag/build_vector_store.py --standards-dir standards/ --db-path rag/qdrant_db
    python rag/build_vector_store.py --rebuild
"""

import argparse
import json
import os
import shutil
from types import MethodType
from pathlib import Path
from typing import List, Dict

from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_text_splitters import MarkdownTextSplitter
from langchain_core.documents import Document


# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
DEFAULT_COLLECTION_NAME = "go_coding_standards"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


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


# ── Document Loading ─────────────────────────────────────────────────────────

def load_markdown_standards(standards_dir: str) -> List[Document]:
    """
    Load all .md files from the standards directory.
    Each markdown file represents a category of coding standards.
    """
    loader = DirectoryLoader(
        standards_dir,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    print(f"[INFO] Loaded {len(documents)} markdown files from {standards_dir}")
    return documents


def load_json_rules(rules_path: str) -> List[Document]:
    """
    Load structured rules from JSON and convert each rule into a Document.
    This gives each rule its own embedding for precise retrieval.
    """
    documents = []

    if not os.path.exists(rules_path):
        print(f"[WARN] Rules file not found: {rules_path}")
        return documents

    with open(rules_path, "r", encoding="utf-8") as f:
        rules = json.load(f)

    for rule in rules:
        # Build rich text representation of the rule for embedding
        content = (
            f"Rule ID: {rule['rule_id']}\n"
            f"Category: {rule['category']}\n"
            f"Severity: {rule['severity']}\n"
            f"Title: {rule['title']}\n\n"
            f"Description: {rule['description']}\n\n"
        )

        if "violation_example" in rule:
            content += f"Violation Example:\n```go\n{rule['violation_example']}\n```\n\n"
        if "correct_example" in rule:
            content += f"Correct Example:\n```go\n{rule['correct_example']}\n```\n"

        doc = Document(
            page_content=content,
            metadata={
                "rule_id": rule["rule_id"],
                "category": rule["category"],
                "severity": rule["severity"],
                "title": rule["title"],
                "source": rules_path,
                "auto_fixable": rule.get("auto_fixable", False),
            },
        )
        documents.append(doc)

    print(f"[INFO] Loaded {len(documents)} individual rules from {rules_path}")
    return documents


# ── Chunking ─────────────────────────────────────────────────────────────────

def chunk_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split markdown documents into smaller chunks for better retrieval.
    JSON rules are not chunked (each rule is already a single document).
    """
    splitter = MarkdownTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks


# ── Embedding & Indexing ─────────────────────────────────────────────────────

def create_embeddings(model_name: str = DEFAULT_EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """Initialize the embedding model."""
    print(f"[INFO] Loading embedding model: {model_name}")

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            "trust_remote_code": True,
            "device": "cpu",  # Embeddings are small, CPU is fine
        },
        encode_kwargs={
            "normalize_embeddings": True,  # For cosine similarity
        },
    )

    return embeddings


def build_vector_store(
    chunks: List[Document],
    embeddings: HuggingFaceEmbeddings,
    db_path: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Qdrant:
    """
    Build Qdrant vector store from document chunks.
    Uses local/on-disk mode (no Qdrant server required).
    """
    print(f"[INFO] Building vector store at {db_path}...")
    print(f"[INFO] Collection: {collection_name}")
    print(f"[INFO] Indexing {len(chunks)} chunks...")

    try:
        client = QdrantClient(path=db_path)
    except RuntimeError as e:
        # Local Qdrant uses a file lock; if another process holds it,
        # provide a clear actionable error.
        if "already accessed by another instance" in str(e):
            raise RuntimeError(
                f"Qdrant local storage is locked at '{db_path}'. "
                "Close other Python processes using this DB, then retry with --rebuild."
            ) from e
        raise

    client = ensure_qdrant_search_compat(client)

    # Recreate collection manually for broad client compatibility.
    try:
        client.delete_collection(collection_name=collection_name)
    except Exception:
        pass

    vector_size = len(embeddings.embed_query("go coding standards"))
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    vector_store = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings,
    )
    vector_store.add_documents(chunks)

    print(f"[INFO] Vector store built successfully with {len(chunks)} entries")
    return vector_store


# ── Verification ─────────────────────────────────────────────────────────────

def verify_vector_store(
    vector_store: Qdrant,
    test_queries: List[str] = None,
) -> None:
    """Run test queries to verify the vector store works correctly."""
    if test_queries is None:
        test_queries = [
            "error handling wrapping context",
            "goroutine leak prevention",
            "hardcoded secrets security",
            "table driven tests",
            "naming convention acronyms",
        ]

    print(f"\n[INFO] Verifying vector store with {len(test_queries)} test queries...")

    for query in test_queries:
        results = vector_store.similarity_search(query, k=3)
        print(f"\n  Query: '{query}'")
        for i, doc in enumerate(results):
            rule_id = doc.metadata.get("rule_id", "N/A")
            category = doc.metadata.get("category", "N/A")
            snippet = doc.page_content[:80].replace("\n", " ")
            print(f"    [{i+1}] {rule_id} ({category}): {snippet}...")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build Qdrant vector store from Go coding standards"
    )
    parser.add_argument(
        "--standards-dir",
        default="standards",
        help="Path to standards documents directory (default: standards/)",
    )
    parser.add_argument(
        "--rules-json",
        default="standards/rules.json",
        help="Path to structured rules JSON (default: standards/rules.json)",
    )
    parser.add_argument(
        "--db-path",
        default="rag/qdrant_db",
        help="Path to store Qdrant database (default: rag/qdrant_db)",
    )
    parser.add_argument(
        "--collection-name",
        default=DEFAULT_COLLECTION_NAME,
        help=f"Qdrant collection name (default: {DEFAULT_COLLECTION_NAME})",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Hugging Face embedding model (default: {DEFAULT_EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Chunk size for text splitting (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete existing vector store and rebuild from scratch",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Run verification queries after building (default: True)",
    )

    args = parser.parse_args()

    # Rebuild if requested
    if args.rebuild and os.path.exists(args.db_path):
        print(f"[INFO] Removing existing vector store at {args.db_path}...")
        shutil.rmtree(args.db_path)

    # Load documents from both sources
    all_documents = []
    
    # 1. Load markdown standards
    if os.path.isdir(args.standards_dir):
        md_docs = load_markdown_standards(args.standards_dir)
        md_chunks = chunk_documents(md_docs, args.chunk_size)
        all_documents.extend(md_chunks)

    # 2. Load individual JSON rules (not chunked — each rule is one doc)
    if os.path.exists(args.rules_json):
        rule_docs = load_json_rules(args.rules_json)
        all_documents.extend(rule_docs)

    if not all_documents:
        print("[ERROR] No documents loaded. Check --standards-dir and --rules-json paths.")
        return

    print(f"\n[INFO] Total documents to index: {len(all_documents)}")

    # Create embeddings
    embeddings = create_embeddings(args.embedding_model)

    # Build vector store
    vector_store = build_vector_store(
        all_documents, embeddings, args.db_path, args.collection_name
    )

    # Verify
    if args.verify:
        verify_vector_store(vector_store)

    print(f"\n[SUCCESS] Vector store ready at {args.db_path}")
    print(f"[INFO] Use retriever.py to query at inference time")


if __name__ == "__main__":
    main()
