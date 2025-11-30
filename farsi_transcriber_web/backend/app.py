"""
Farsi Transcriber Backend API

Flask API for handling audio/video file transcription using Whisper model.
Configured for Railway deployment with async job processing.
"""

import os
import sys
import tempfile
import uuid
import threading
import concurrent.futures
import time
from pathlib import Path
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify
from flask_cors import CORS

# Prevent model download during build
os.environ['WHISPER_CACHE'] = os.path.expanduser('~/.cache/whisper')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from farsi_transcriber.core.transcriber import FarsiTranscriber
from farsi_transcriber.core.export import TranscriptionExporter

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configuration
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a', 'flac', 'ogg', 'aac', 'wma', 'mp4', 'mkv', 'mov', 'webm', 'avi', 'flv', 'wmv'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

# Production settings
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
app.config['ENV'] = os.getenv('FLASK_ENV', 'production')

# Job Management
jobs = {}
executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

class JobManager:
    @staticmethod
    def start_job(file_path, language='fa'):
        job_id = str(uuid.uuid4())
        jobs[job_id] = {
            'id': job_id,
            'status': 'pending',
            'progress': 0,
            'result': None,
            'error': None,
            'filename': Path(file_path).name,
            'submitted_at': time.time()
        }
        executor.submit(JobManager.process_job, job_id, file_path, language)
        return job_id

    @staticmethod
    def process_job(job_id, file_path, language):
        try:
            jobs[job_id]['status'] = 'processing'

            # Lazy load model (per thread if needed, but Whisper loads globally mostly)
            # We initialize a new Transcriber which loads the model
            # Note: In a real prod env, we'd want a dedicated worker process keeping the model in memory.
            # Here we rely on Whisper's caching or global state if possible, but FarsiTranscriber inits it.
            # To avoid reloading model every time, we might want to cache the transcriber instance globally.
            transcriber = get_transcriber()

            # Since FarsiTranscriber doesn't have a callback for progress,
            # we can't easily update percentage accurately without hacking Whisper.
            # We will simulate progress or just stay at "processing".

            # Transcribe
            result = transcriber.transcribe(file_path, language=language)

            # Enhance result with full text if not present (FarsiTranscriber does this but let's be safe)
            if 'full_text' not in result:
                result['full_text'] = result.get('text', '')

            jobs[job_id]['result'] = result
            jobs[job_id]['status'] = 'completed'
            jobs[job_id]['progress'] = 100

        except Exception as e:
            jobs[job_id]['status'] = 'error'
            jobs[job_id]['error'] = str(e)
        finally:
            # Cleanup file
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except:
                pass

# Global Transcriber Instance (Lazy Loaded)
_transcriber_instance = None
_transcriber_lock = threading.Lock()

def get_transcriber():
    global _transcriber_instance
    with _transcriber_lock:
        if _transcriber_instance is None:
            # Use medium model as default
            _transcriber_instance = FarsiTranscriber(model_name="medium")
    return _transcriber_instance

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    """Root endpoint"""
    return jsonify({
        'message': 'Farsi Transcriber API',
        'version': '2.0.0',
        'status': 'running'
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'active_jobs': len([j for j in jobs.values() if j['status'] in ['pending', 'processing']]),
        'environment': app.config['ENV']
    })

@app.route('/api/jobs', methods=['POST'])
def create_job():
    """Upload file and start transcription job"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    # Save file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    language = request.form.get('language', 'fa')

    job_id = JobManager.start_job(filepath, language)

    return jsonify({
        'status': 'success',
        'job_id': job_id,
        'message': 'Job started successfully'
    })

@app.route('/api/jobs/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get job status"""
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    response = {
        'id': job['id'],
        'status': job['status'],
        'progress': job['progress'],
        'filename': job['filename'],
        'submitted_at': job['submitted_at']
    }

    if job['status'] == 'error':
        response['error'] = job['error']

    if job['status'] == 'completed':
        # Don't send full result here to keep it light, just summary
        response['segment_count'] = len(job['result'].get('segments', []))

    return jsonify(response)

@app.route('/api/jobs/<job_id>/result', methods=['GET'])
def get_job_result(job_id):
    """Get full job result"""
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    if job['status'] != 'completed':
        return jsonify({'error': 'Job not completed'}), 400

    return jsonify(job['result'])

@app.route('/api/export/<job_id>', methods=['GET'])
def export_job(job_id):
    """Export job result in specific format"""
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    if job['status'] != 'completed':
        return jsonify({'error': 'Job not completed'}), 400

    format_type = request.args.get('format', 'txt').lower()

    # Use temporary file to generate export
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        TranscriptionExporter.export(job['result'], tmp_path, format_type)
        with open(tmp_path, 'r', encoding='utf-8') as f:
            content = f.read()

        mime_types = {
            'txt': 'text/plain',
            'srt': 'text/plain',
            'vtt': 'text/plain',
            'json': 'application/json',
            'tsv': 'text/tab-separated-values'
        }

        return jsonify({
            'status': 'success',
            'format': format_type,
            'content': content,
            'mime_type': mime_types.get(format_type, 'text/plain')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV', 'production') == 'development'
    app.run(debug=debug, host='0.0.0.0', port=port, threaded=True)
