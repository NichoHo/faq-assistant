# file name: rag_pipeline.py
import os
from typing import List, Tuple

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Configuration
PERSIST_DIR = "vectorstore"

# Global cache
_embeddings = None
_vectordb = None

def get_embeddings():
    """Get or create embeddings model."""
    global _embeddings
    if _embeddings is None:
        print("🔧 Loading embeddings model...")
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        print("✅ Embeddings model loaded.")
    return _embeddings

def get_vectorstore(persist_directory: str = PERSIST_DIR):
    """Get or create vector store."""
    global _vectordb
    if _vectordb is None:
        print(f"📚 Loading FAISS vector store from: {persist_directory}")
        embeddings = get_embeddings()
        try:
            # Try without the dangerous deserialization parameter first
            _vectordb = FAISS.load_local(folder_path=persist_directory, embeddings=embeddings, allow_dangerous_deserialization=True)
        except TypeError as e:
            # If that fails, try with the parameter (for newer versions)
            if "allow_dangerous_deserialization" in str(e):
                _vectordb = FAISS.load_local(
                    folder_path=persist_directory,
                    embeddings=embeddings,
                    allow_dangerous_deserialization=True
                )
            else:
                raise
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
    
    # Loop through each unique document chunk found
    for i, doc in enumerate(docs, 1):
        # Extract metadata
        source_name = os.path.basename(doc.metadata.get('source', 'Unknown File'))
        page_num = doc.metadata.get('page', 'N/A')
        
        # Clean up the content (replace newlines with spaces to avoid broken sentences, 
        # but keep structure clean)
        content = " ".join(doc.page_content.split())
        
        # Construct a formatted block using Markdown
        # We use \n\n to ensure the frontend renders a new paragraph
        entry = (
            f"**Excerpt from {source_name} (Page {page_num}):**\n\n"
            f"{content}"
        )
        formatted_answers.append(entry)

    # Join all separate parts with a horizontal rule or spacing for visual separation
    return "\n\n---\n\n".join(formatted_answers)

def answer_question(
    query: str,
    k: int = 4,
    return_docs: bool = False,
    persist_directory: str = PERSIST_DIR
) -> Tuple[str, List[Document]]:
    """RAG pipeline to answer a question."""
    try:
        # Check if vector store exists
        if not os.path.exists(persist_directory):
            error_msg = "Vector store not found. Please upload and process PDFs first."
            return error_msg, [] if not return_docs else []
        
        # Get vector store
        vectordb = get_vectorstore(persist_directory)
        
        # Retrieve relevant chunks
        # We fetch k=3 to keep the answer concise but useful
        retrieved_docs = vectordb.similarity_search(query, k=3)
        
        if not retrieved_docs:
            answer = "No relevant information found in the policy documents."
            return answer, [] if not return_docs else []
        
        # Generate answer using the new non-destructive method
        answer = simple_answer(retrieved_docs, query)
        
        if return_docs:
            return answer, retrieved_docs
        else:
            return answer, []
            
    except Exception as e:
        error_msg = f"Error generating answer: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg, [] if not return_docs else []

def check_vectorstore_ready(persist_directory: str = PERSIST_DIR) -> bool:
    """Check if vector store is ready."""
    return os.path.exists(persist_directory) and any(
        fname.endswith('.pkl') or fname.endswith('.faiss') 
        for fname in os.listdir(persist_directory)
    )

if __name__ == "__main__":
    # Test the pipeline
    print("🧪 Testing RAG pipeline...")
    
    if not check_vectorstore_ready():
        print("❌ Vector store not ready. Please process PDFs first.")
    else:
        test_questions = [
            "What is the dress code?",
        ]
        
        for question in test_questions:
            print(f"\n{'='*60}")
            print(f"Question: {question}")
            answer, docs = answer_question(question, k=3, return_docs=True)
            print(f"Answer:\n{answer}")