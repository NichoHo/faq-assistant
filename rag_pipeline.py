import os
from typing import List, Tuple

from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

from embeddings import get_embeddings

_vectordb = None

def reset_vectorstore_cache() -> None:
    global _vectordb
    _vectordb = None


def get_vectorstore():
    """Get or create Pinecone vector store."""
    global _vectordb

    if _vectordb is None:
        print("📚 Connecting to Pinecone vector store...")
        embeddings = get_embeddings()
        index_name = "faq-assistant"
        _vectordb = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings
        )
        print("✅ Connected to Pinecone.")
    return _vectordb


def simple_answer(docs: List[Document], question: str) -> str:
    if not docs:
        return "I couldn't find any information about that in the uploaded documents."

    formatted_answers = []
    for i, doc in enumerate(docs, 1):
        source_name = os.path.basename(doc.metadata.get("source", "Unknown File"))
        page_num = doc.metadata.get("page", "N/A")
        content = " ".join(doc.page_content.split())

        entry = (
            f"**Excerpt from {source_name} (Page {page_num}):**\n\n"
            f"{content}"
        )
        formatted_answers.append(entry)

    return "\n\n---\n\n".join(formatted_answers)


def answer_question(
    query: str,
    k: int = 4,
    return_docs: bool = False,
    persist_directory: str = None,
) -> Tuple[str, List[Document]]:
    """RAG pipeline to answer a question using Pinecone."""
    try:
        vectordb = get_vectorstore()
        retrieved_docs = vectordb.similarity_search(query, k=k)

        if not retrieved_docs:
            answer = "No relevant information found in the policy documents."
            return answer, [] if not return_docs else []

        answer = simple_answer(retrieved_docs, query)

        if return_docs:
            return answer, retrieved_docs
        return answer, []

    except Exception as e:
        error_msg = f"Error generating answer: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg, [] if not return_docs else []