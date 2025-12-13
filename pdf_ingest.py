import os

# Updated imports for modular LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Changed
from langchain_huggingface.embeddings import HuggingFaceEmbeddings    # Also updated
from langchain_community.vectorstores import FAISS

PDF_DIR = "data/pdfs"
PERSIST_DIR = "vectorstore"

# Create directories if they don't exist
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

def load_pdfs(pdf_dir: str):
    """Load all PDF files from directory."""
    print(f"📄 Loading PDFs from: {pdf_dir}")
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
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

def chunk_documents(docs, chunk_size=1000, chunk_overlap=200):
    """Split documents into chunks."""
    print("🔪 Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ Created {len(chunks)} chunks.")
    return chunks

def build_vectorstore(chunks, persist_directory=PERSIST_DIR):
    """Create embeddings and build FAISS vector store."""
    print("🤖 Creating embeddings model: all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    print(f"💾 Building FAISS vector store at: {persist_directory}")
    vectordb = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    
    # Save the vector store
    vectordb.save_local(persist_directory)
    print("✅ Vector store built and saved.")
    return vectordb

def get_pdf_count(pdf_dir: str = PDF_DIR):
    """Count PDF files in directory."""
    if not os.path.isdir(pdf_dir):
        return 0
    return len([f for f in os.listdir(pdf_dir) if f.endswith('.pdf')])

def process_pdfs(pdf_dir: str = PDF_DIR, persist_directory: str = PERSIST_DIR):
    """Main function to process PDFs and build vector store."""
    try:
        # Create directories if they don't exist
        os.makedirs(pdf_dir, exist_ok=True)
        os.makedirs(persist_directory, exist_ok=True)
        
        pdf_count = get_pdf_count(pdf_dir)
        if pdf_count == 0:
            return False, f"No PDF files found in '{pdf_dir}'."
        
        print(f"📊 Found {pdf_count} PDF file(s) to process.")
        
        # Load, chunk, and build vector store
        docs = load_pdfs(pdf_dir)
        chunks = chunk_documents(docs)
        vectordb = build_vectorstore(chunks, persist_directory)
        
        return True, f"Successfully processed {pdf_count} PDF(s) and created vector store with {len(chunks)} chunks."
        
    except Exception as e:
        return False, f"Error processing PDFs: {str(e)}"

def main():
    """Command-line entry point."""
    success, message = process_pdfs()
    
    if success:
        print(f"✅ {message}")
        return 0
    else:
        print(f"❌ {message}")
        return 1

if __name__ == "__main__":
    exit(main())