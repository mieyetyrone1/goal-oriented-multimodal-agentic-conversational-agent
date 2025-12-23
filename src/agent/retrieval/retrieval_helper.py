from typing import List, Dict, Tuple
import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def build_retrieval_context(
    query: str,
    documents: List[Dict],
    embedding_model,
    top_k: int = 3,
    score_threshold: float = 0.6,
) -> str:
    """
    Build a retrieval-augmented context string using embedding similarity.
    Returns an empty string if no relevant documents are found.
    """

    if not documents:
        return ""

    # Embed query
    query_embedding = embedding_model.embed([query])[0]

    # Embed documents
    doc_texts = [doc["content"] for doc in documents]
    doc_embeddings = embedding_model.embed(doc_texts)
    
    # Score documents + filter documents
    scored_docs: List[Tuple[float, Dict]] = []
    for doc, emb in zip(documents, doc_embeddings):
        score = cosine_similarity(query_embedding, emb)
        if score >= score_threshold:
            scored_docs.append((score, doc))

    # No relevant documents
    if not scored_docs:
        return ""

    # Sort by relevance
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    # Select top-K
    top_docs = scored_docs[:top_k]

    # Build context prompt without citation.
    """context = (
        "Use the following information to answer the user's question.\n"
        "If the information is insufficient, say so.\n\n"
    )"""
    # Build context with citation instructions
    context = (
        "You are answering using the sources below.\n"
        "Cite facts using bracketed numbers like [1], [2].\n"
        "If the sources do not contain the answer, say so.\n\n"
        "SOURCES:\n"
    )

    for i, (score, doc) in enumerate(top_docs, 1):
        context += f"[{i}] (score: {score:.2f}) {doc['content']}\n"

    return context
