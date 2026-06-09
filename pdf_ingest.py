import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

from config import PDF_DIR, get_pdf_source_dir, ensure_runtime_dirs
from embeddings import get_embeddings

ensure_runtime_dirs()


def load_pdfs(pdf_dir: str):
    """Load all PDF files from directory."""
    print(f"📄 Loading PDFs from: {pdf_dir}")
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {pdf_dir}")

    documents = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        print(f"  Loading: {pdf_file}")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = pdf_file
        documents.extend(docs)

    print(f"✅ Loaded {len(documents)} pages from {len(pdf_files)} PDF file(s).")
    return documents


def chunk_documents(docs, chunk_size=3000, chunk_overlap=300):
    """Split documents into chunks."""
    print("🔪 Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ Created {len(chunks)} chunks.")
    return chunks


def build_vectorstore(chunks, persist_directory=None):
    """Create embeddings and build Pinecone vector store."""
    print("🤖 Creating embeddings model: all-MiniLM-L6-v2")
    embeddings = get_embeddings()

    index_name = "faq-assistant" # Must match your Pinecone index name
    print(f"☁️ Uploading chunks to Pinecone index: {index_name}")
    
    vectordb = PineconeVectorStore.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        index_name=index_name
    )
    print("✅ Vector store built and saved to Pinecone.")
    return vectordb


def get_pdf_count(pdf_dir: str = PDF_DIR):
    """Count PDF files in directory."""
    if not os.path.isdir(pdf_dir):
        return 0
    return len([f for f in os.listdir(pdf_dir) if f.endswith(".pdf")])


def process_pdfs(pdf_dir: str = None, persist_directory: str = None):
    """Main function to process PDFs and build vector store."""
    pdf_dir = pdf_dir or get_pdf_source_dir()

    try:
        os.makedirs(pdf_dir, exist_ok=True)
        pdf_count = get_pdf_count(pdf_dir)
        
        if pdf_count == 0:
            return False, f"No PDF files found in '{pdf_dir}'."

        print(f"📊 Found {pdf_count} PDF file(s) to process.")

        docs = load_pdfs(pdf_dir)
        chunks = chunk_documents(docs)
        build_vectorstore(chunks)

        return True, (
            f"Successfully processed {pdf_count} PDF(s) and "
            f"uploaded {len(chunks)} chunks to Pinecone."
        )

    except Exception as e:
        return False, f"Error processing PDFs: {str(e)}"


def main():
    success, message = process_pdfs()
    if success:
        print(f"✅ {message}")
        return 0
    print(f"❌ {message}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())