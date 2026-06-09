import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IS_VERCEL = os.environ.get("VERCEL") == "1"

# On Vercel, only /tmp is writable at runtime. During build, the project dir is writable.
DATA_ROOT = os.environ.get(
    "FAQ_DATA_ROOT",
    "/tmp/faq-assistant" if IS_VERCEL else BASE_DIR,
)

PDF_DIR = os.environ.get(
    "PDF_DIR",
    os.path.join(DATA_ROOT, "data", "pdfs"),
)

RUNTIME_VECTORSTORE_DIR = os.environ.get(
    "VECTORSTORE_DIR",
    os.path.join(DATA_ROOT, "vectorstore"),
)

# Vector store produced at build time and bundled with the deployment.
BUNDLED_VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")

# PDFs committed for build-time indexing (optional).
DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")


def _has_vectorstore_files(directory: str) -> bool:
    if not os.path.isdir(directory):
        return False
    return any(
        fname.endswith(".pkl") or fname.endswith(".faiss")
        for fname in os.listdir(directory)
    )


def get_vectorstore_dir() -> str:
    """Prefer bundled vector store, then runtime-writable store."""
    if _has_vectorstore_files(BUNDLED_VECTORSTORE_DIR):
        return BUNDLED_VECTORSTORE_DIR
    return RUNTIME_VECTORSTORE_DIR


def get_pdf_source_dir() -> str:
    """Use committed documents/ for build-time indexing when present."""
    if os.path.isdir(DOCUMENTS_DIR) and any(
        f.endswith(".pdf") for f in os.listdir(DOCUMENTS_DIR)
    ):
        return DOCUMENTS_DIR
    return PDF_DIR


def ensure_runtime_dirs() -> None:
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(RUNTIME_VECTORSTORE_DIR, exist_ok=True)


def is_upload_enabled() -> bool:
    """Upload + re-index is disabled on Vercel when a bundled vector store exists."""
    if not IS_VERCEL:
        return True
    return not _has_vectorstore_files(BUNDLED_VECTORSTORE_DIR)
