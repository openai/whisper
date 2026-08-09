"""
Farsi Transcriber Backend API

Flask API for handling audio/video file transcription using Whisper model.
Configured for Railway deployment with lazy model loading.
"""

import os
import sys
import tempfile
from pathlib import Path
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify
from flask_cors import CORS

# Prevent model download during build
os.environ['WHISPER_CACHE'] = os.path.expanduser('~/.cache/whisper')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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

# Load Whisper model (lazy load - only on first transcription request)
model = None

def load_model():
    """Lazy load Whisper model on first use (not during build)"""
    global model
    if model is None:
        try:
            print("⏳ Loading Whisper model for first time...")
            print("   This may take 1-2 minutes on first run...")
            # Import here to avoid loading during build
            import whisper
            model = whisper.load_model('medium')
            print("✓ Whisper model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading Whisper model: {e}")
            model = None
    return model


def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
def index():
    """Root endpoint"""
    return jsonify({
        'message': 'Farsi Transcriber API',
        'version': '1.0.0',
        'status': 'running'
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint - fast response without loading model"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'environment': app.config['ENV']
    })


@app.route('/transcribe', methods=['POST'])
def transcribe():
    """
    Transcribe audio/video file

    Request:
    - file: Audio/video file
    - language: Language code (default: 'fa' for Farsi)

    Response:
    - transcription results with segments and timestamps
    """
    try:
        # Load model if not already loaded
        whisper_model = load_model()
        if not whisper_model:
            return jsonify({'error': 'Failed to load Whisper model'}), 500

        # Check if file is in request
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

        # Get language code from request (default: Farsi)
        language = request.form.get('language', 'fa')

        # Transcribe
        result = whisper_model.transcribe(filepath, language=language, verbose=False)

        # Format response
        segments = []
        for segment in result.get('segments', []):
            segments.append({
                'start': f"{int(segment['start'] // 3600):02d}:{int((segment['start'] % 3600) // 60):02d}:{int(segment['start'] % 60):02d}.{int((segment['start'] % 1) * 1000):03d}",
                'end': f"{int(segment['end'] // 3600):02d}:{int((segment['end'] % 3600) // 60):02d}:{int(segment['end'] % 60):02d}.{int((segment['end'] % 1) * 1000):03d}",
                'text': segment['text'].strip(),
            })

        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass

        return jsonify({
            'status': 'success',
            'filename': filename,
            'language': result.get('language', 'unknown'),
            'text': result.get('text', ''),
            'segments': segments
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/models', methods=['GET'])
def get_models():
    """Get available Whisper models"""
    return jsonify({
        'available_models': ['tiny', 'base', 'small', 'medium', 'large'],
        'current_model': 'medium',
        'description': 'List of available Whisper models. Larger models are more accurate but slower.'
    })


@app.route('/export', methods=['POST'])
def export():
    """
    Export transcription in specified format

    Request:
    - transcription: Full transcription text
    - segments: Array of segments with timestamps
    - format: Export format (txt, srt, vtt, json)

    Response:
    - Exported file content
    """
    try:
        data = request.json
        transcription = data.get('transcription', '')
        segments = data.get('segments', [])
        format_type = data.get('format', 'txt').lower()

        if format_type == 'txt':
            content = transcription
            mime_type = 'text/plain'
        elif format_type == 'srt':
            content = _format_srt(segments)
            mime_type = 'text/plain'
        elif format_type == 'vtt':
            content = _format_vtt(segments)
            mime_type = 'text/plain'
        elif format_type == 'json':
            import json
            content = json.dumps({'text': transcription, 'segments': segments}, ensure_ascii=False, indent=2)
            mime_type = 'application/json'
        else:
            return jsonify({'error': 'Unsupported format'}), 400

        return jsonify({
            'status': 'success',
            'format': format_type,
            'content': content,
            'mime_type': mime_type
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _format_srt(segments):
    """Format transcription as SRT subtitle format"""
    lines = []
    for i, segment in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{segment['start']} --> {segment['end']}")
        lines.append(segment['text'])
        lines.append('')
    return '\n'.join(lines)


def _format_vtt(segments):
    """Format transcription as WebVTT subtitle format"""
    lines = ['WEBVTT', '']
    for segment in segments:
        lines.append(f"{segment['start']} --> {segment['end']}")
        lines.append(segment['text'])
        lines.append('')
    return '\n'.join(lines)


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV', 'production') == 'development'
    app.run(debug=debug, host='0.0.0.0', port=port, threaded=True)
