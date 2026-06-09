from flask import Flask, render_template, request, jsonify
import os
import vercel_blob

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

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        from werkzeug.utils import secure_filename

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        
        # Save locally to /tmp for PyPDFLoader to read
        file.save(file_path)

        try:
            # Back up the file to Vercel Blob
            with open(file_path, "rb") as f:
                vercel_blob.put(filename, f.read(), options={"access": "public"})
                
            from pdf_ingest import process_pdfs

            # Process and send to Pinecone
            success, message = process_pdfs()
            if success:
                from rag_pipeline import reset_vectorstore_cache
                reset_vectorstore_cache()
                return jsonify({"message": f"File {filename} successfully saved to cloud and indexed!"})
            
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
        from rag_pipeline import answer_question
        answer, docs = answer_question(question, k=4, return_docs=True)

        sources = []
        for i, doc in enumerate(docs, 1):
            sources.append({
                "id": i,
                "source": os.path.basename(doc.metadata.get("source", "Unknown")),
                "page": doc.metadata.get("page", "N/A"),
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            })

        return jsonify({
            "question": question,
            "answer": answer,
            "sources": sources,
        })

    except Exception as e:
        return jsonify({"error": f"Error getting answer: {str(e)}"}), 500

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