import os
from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import get_vectorstore_dir
from embeddings import get_embeddings

_vectordb = None


def reset_vectorstore_cache() -> None:
    global _vectordb
    _vectordb = None


def get_vectorstore(persist_directory: str = None):
    """Get or create vector store."""
    global _vectordb
    persist_directory = persist_directory or get_vectorstore_dir()

    if _vectordb is None:
        print(f"📚 Loading FAISS vector store from: {persist_directory}")
        embeddings = get_embeddings()
        _vectordb = FAISS.load_local(
            folder_path=persist_directory,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        print("✅ Vector store loaded.")
    return _vectordb


def simple_answer(docs: List[Document], question: str) -> str:
    """
    Generates an answer by formatting the retrieved documents nicely.
    Since we don't have a real LLM here, we present the relevant excerpts clearly.
    """
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
    """RAG pipeline to answer a question."""
    persist_directory = persist_directory or get_vectorstore_dir()

    try:
        if not os.path.exists(persist_directory):
            error_msg = "Vector store not found. Please upload and process PDFs first."
            return error_msg, [] if not return_docs else []

        vectordb = get_vectorstore(persist_directory)
        retrieved_docs = vectordb.similarity_search(query, k=3)

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


def check_vectorstore_ready(persist_directory: str = None) -> bool:
    """Check if vector store is ready."""
    persist_directory = persist_directory or get_vectorstore_dir()
    return os.path.exists(persist_directory) and any(
        fname.endswith(".pkl") or fname.endswith(".faiss")
        for fname in os.listdir(persist_directory)
    )


if __name__ == "__main__":
    print("🧪 Testing RAG pipeline...")

    if not check_vectorstore_ready():
        print("❌ Vector store not ready. Please process PDFs first.")
    else:
        test_questions = ["What is the dress code?"]

        for question in test_questions:
            print(f"\n{'=' * 60}")
            print(f"Question: {question}")
            answer, docs = answer_question(question, k=3, return_docs=True)
            print(f"Answer:\n{answer}")
