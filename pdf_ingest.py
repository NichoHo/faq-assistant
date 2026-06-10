import os
from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from embeddings import get_embeddings

def process_pdfs(file_path: str, session_id: str):
    try:
        print(f"📄 Processing file for session {session_id}")
        
        # 1. Delete any existing data in this user's session namespace
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        index = pc.Index("faq-assistant")
        try:
            index.delete(delete_all=True, namespace=session_id)
        except Exception:
            pass # Namespace might be completely new, which is fine

        # 2. Load the specific PDF
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = os.path.basename(file_path)

        # 3. Chunk the text
        splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
        chunks = splitter.split_documents(docs)

        # 4. Upload to Pinecone strictly within the user's namespace
        PineconeVectorStore.from_documents(
            documents=chunks, 
            embedding=get_embeddings(), 
            index_name="faq-assistant",
            namespace=session_id
        )

        # 5. Delete the file from the Render server to save disk space
        if os.path.exists(file_path):
            os.remove(file_path)

        return True, f"Successfully processed and indexed."

    except Exception as e:
        return False, f"Error processing PDF: {str(e)}"