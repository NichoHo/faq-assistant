from flask import Flask, render_template, request, jsonify
import os

from config import (
    PDF_DIR,
    ensure_runtime_dirs,
    is_upload_enabled,
)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = PDF_DIR
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

ensure_runtime_dirs()

ALLOWED_EXTENSIONS = {"pdf"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    session_id = request.form.get("session_id", "default_session") # Get session ID

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        from werkzeug.utils import secure_filename
        import time

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{int(time.time())}_{filename}")
        file.save(file_path)

        try:
            from pdf_ingest import process_pdfs
            success, message = process_pdfs(file_path, session_id) # Pass to ingest
            
            if success:
                return jsonify({"message": "File indexed successfully!"})
            return jsonify({"error": message}), 500
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Invalid file type."}), 400

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()
    question = data.get("question", "").strip()
    session_id = data.get("session_id", "default_session") # Get session ID

    if not question:
        return jsonify({"error": "Please enter a question"}), 400

    try:
        from rag_pipeline import answer_question
        answer, docs = answer_question(question, session_id, k=4, return_docs=True) # Pass to RAG

        return jsonify({
            "question": question,
            "answer": answer,
            "sources": []
        })
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500

@app.route("/status")
def check_status():
    # Since we use Pinecone, the vector database is always "Ready" in the cloud
    return jsonify({
        "vectorstore_ready": True, 
        "pdfs_uploaded": True,
        "pdf_count": "Cloud",
        "upload_enabled": True,
        "deployment": "vercel" if os.environ.get("VERCEL") == "1" else "local",
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)