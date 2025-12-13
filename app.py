from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import shutil
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data/pdfs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('vectorstore', exist_ok=True)

ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process the PDF
        try:
            from pdf_ingest import process_pdfs
            # Check if there are PDFs to process
            import os
            pdf_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith('.pdf')]
            if not pdf_files:
                return jsonify({'error': 'No PDF files found to process'}), 400
                
            success, message = process_pdfs()
            if success:
                return jsonify({'message': f'File {filename} uploaded and processed successfully!'})
            else:
                return jsonify({'error': f'Error processing PDF: {message}'}), 500
        except Exception as e:
            return jsonify({'error': f'Error processing PDF: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type. Only PDF files are allowed.'}), 400

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({'error': 'Please enter a question'}), 400
    
    try:
        # Check if vector store exists
        if not os.path.exists('vectorstore') or not os.listdir('vectorstore'):
            return jsonify({'error': 'No processed documents found. Please upload a PDF first.'}), 400
        
        # Get answer from RAG pipeline
        from rag_pipeline import answer_question
        answer, docs = answer_question(question, k=4, return_docs=True)
        
        # Format sources
        sources = []
        for i, doc in enumerate(docs, 1):
            sources.append({
                'id': i,
                'source': os.path.basename(doc.metadata.get('source', 'Unknown')),
                'page': doc.metadata.get('page', 'N/A'),
                'content': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content
            })
        
        return jsonify({
            'question': question,
            'answer': answer,
            'sources': sources
        })
    
    except Exception as e:
        return jsonify({'error': f'Error getting answer: {str(e)}'}), 500

@app.route('/status')
def check_status():
    # Check if vector store exists and has data
    vectorstore_exists = os.path.exists('vectorstore') and any(fname.endswith('.pkl') or fname.endswith('.faiss') for fname in os.listdir('vectorstore'))
    pdfs_exist = os.path.exists('data/pdfs') and os.listdir('data/pdfs')
    
    return jsonify({
        'vectorstore_ready': vectorstore_exists,
        'pdfs_uploaded': pdfs_exist,
        'pdf_count': len([f for f in os.listdir('data/pdfs') if f.endswith('.pdf')]) if pdfs_exist else 0
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)