from flask import Flask, render_template, request, jsonify
import os

from config import (
    PDF_DIR,
    ensure_runtime_dirs,
    get_vectorstore_dir,
    is_upload_enabled,
)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = PDF_DIR
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

ensure_runtime_dirs()

ALLOWED_EXTENSIONS = {"pdf"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def vectorstore_is_ready() -> bool:
    vectorstore_dir = get_vectorstore_dir()
    if not os.path.isdir(vectorstore_dir):
        return False
    return any(
        fname.endswith(".pkl") or fname.endswith(".faiss")
        for fname in os.listdir(vectorstore_dir)
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if not is_upload_enabled():
        return jsonify(
            {
                "error": (
                    "Upload is disabled on this deployment. "
                    "Add PDFs to the documents/ folder and redeploy, "
                    "or run the app locally to upload files."
                )
            }
        ), 403

    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        from werkzeug.utils import secure_filename

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        try:
            from pdf_ingest import process_pdfs

            pdf_files = [
                f
                for f in os.listdir(app.config["UPLOAD_FOLDER"])
                if f.endswith(".pdf")
            ]
            if not pdf_files:
                return jsonify({"error": "No PDF files found to process"}), 400

            success, message = process_pdfs()
            if success:
                from rag_pipeline import reset_vectorstore_cache

                reset_vectorstore_cache()
                return jsonify(
                    {"message": f"File {filename} uploaded and processed successfully!"}
                )
            return jsonify({"error": f"Error processing PDF: {message}"}), 500
        except Exception as e:
            return jsonify({"error": f"Error processing PDF: {str(e)}"}), 500

    return jsonify({"error": "Invalid file type. Only PDF files are allowed."}), 400


@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Please enter a question"}), 400

    try:
        if not vectorstore_is_ready():
            return jsonify(
                {
                    "error": (
                        "No processed documents found. "
                        "Add PDFs to documents/ and redeploy, or upload a PDF first."
                    )
                }
            ), 400

        from rag_pipeline import answer_question

        answer, docs = answer_question(question, k=4, return_docs=True)

        sources = []
        for i, doc in enumerate(docs, 1):
            sources.append(
                {
                    "id": i,
                    "source": os.path.basename(doc.metadata.get("source", "Unknown")),
                    "page": doc.metadata.get("page", "N/A"),
                    "content": doc.page_content[:200] + "..."
                    if len(doc.page_content) > 200
                    else doc.page_content,
                }
            )

        return jsonify(
            {
                "question": question,
                "answer": answer,
                "sources": sources,
            }
        )

    except Exception as e:
        return jsonify({"error": f"Error getting answer: {str(e)}"}), 500


@app.route("/status")
def check_status():
    vectorstore_dir = get_vectorstore_dir()
    vectorstore_exists = vectorstore_is_ready()
    pdfs_exist = os.path.isdir(PDF_DIR) and os.listdir(PDF_DIR)

    return jsonify(
        {
            "vectorstore_ready": vectorstore_exists,
            "pdfs_uploaded": pdfs_exist,
            "pdf_count": len([f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")])
            if pdfs_exist
            else 0,
            "upload_enabled": is_upload_enabled(),
            "deployment": "vercel" if os.environ.get("VERCEL") == "1" else "local",
        }
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
