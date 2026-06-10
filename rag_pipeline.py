import os
from typing import List, Tuple
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from embeddings import get_embeddings

def simple_answer(docs: List[Document], question: str) -> str:
    if not docs:
        return "I couldn't find any information about that in your uploaded document."

    formatted_answers = []
    for i, doc in enumerate(docs, 1):
        source_name = os.path.basename(doc.metadata.get("source", "Unknown File"))
        page_num = doc.metadata.get("page", "N/A")
        content = " ".join(doc.page_content.split())
        formatted_answers.append(f"**Excerpt from {source_name} (Page {page_num}):**\n\n{content}")

    return "\n\n---\n\n".join(formatted_answers)

def answer_question(query: str, session_id: str, k: int = 4, return_docs: bool = False) -> Tuple[str, List[Document]]:
    try:
        # Connect to Pinecone and lock into the user's namespace
        vectordb = PineconeVectorStore(
            index_name="faq-assistant",
            embedding=get_embeddings(),
            namespace=session_id
        )
        
        retrieved_docs = vectordb.similarity_search(query, k=k)

        if not retrieved_docs:
            answer = "No relevant information found in the uploaded document."
            return answer, [] if not return_docs else []

        answer = simple_answer(retrieved_docs, query)

        if return_docs:
            return answer, retrieved_docs
        return answer, []

    except Exception as e:
        error_msg = f"Error generating answer: {str(e)}"
        return error_msg, [] if not return_docs else []