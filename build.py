"""Build-time indexing for Vercel deployments."""
import os
import sys

from config import DOCUMENTS_DIR, BASE_DIR


def main() -> int:
    pdf_files = []
    if os.path.isdir(DOCUMENTS_DIR):
        pdf_files = [f for f in os.listdir(DOCUMENTS_DIR) if f.endswith(".pdf")]

    if not pdf_files:
        print("No PDFs in documents/ — skipping vector store build.")
        print("Add PDFs to documents/ and redeploy, or upload locally.")
        return 0

    os.environ["FAQ_DATA_ROOT"] = BASE_DIR

    from pdf_ingest import process_pdfs

    success, message = process_pdfs(
        pdf_dir=DOCUMENTS_DIR,
        persist_directory=os.path.join(BASE_DIR, "vectorstore"),
    )
    print(message)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
