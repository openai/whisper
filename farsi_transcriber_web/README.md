# Farsi Transcriber - Web Application

A professional web-based application for transcribing Farsi audio and video files using OpenAI's Whisper model.

## Features

✨ **Core Features**
- 🎙️ Transcribe audio files (MP3, WAV, M4A, FLAC, OGG, AAC, WMA)
- 🎬 Extract audio from video files (MP4, MKV, MOV, WebM, AVI, FLV, WMV)
- 🇮🇷 High-accuracy Farsi/Persian language transcription
- ⏱️ Word-level timestamps for precise timing
- 📤 Export to multiple formats (TXT, SRT, VTT, JSON)
- 💻 Clean, intuitive React-based UI with Figma design
- 🎨 Dark/Light theme toggle
- 🔍 Search and text highlighting in transcriptions
- 📋 File queue management
- 💾 Copy individual transcription segments
- 🚀 GPU acceleration support (CUDA)
- 🎯 Resizable window for flexible workspace

## Tech Stack

**Frontend:**
- React 18+ with TypeScript
- Vite (fast build tool)
- Tailwind CSS v4.0
- Lucide React (icons)
- re-resizable (window resizing)
- Sonner (toast notifications)

**Backend:**
- Flask (Python web framework)
- OpenAI Whisper (speech recognition)
- PyTorch (deep learning)
- Flask-CORS (cross-origin requests)

## System Requirements

**Frontend:**
- Node.js 16+
- npm/yarn/pnpm

**Backend:**
- Python 3.8+
- 4GB RAM minimum
- 8GB+ recommended
- ffmpeg installed
- Optional: NVIDIA GPU with CUDA support

## Installation

### Step 1: Install ffmpeg

Choose your operating system:

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt install ffmpeg
```

**macOS (Homebrew):**
```bash
brew install ffmpeg
```

**Windows (Chocolatey):**
```bash
choco install ffmpeg
```

### Step 2: Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Frontend Setup

```bash
# Navigate to root directory
cd ..

# Install Node dependencies
npm install

# Or use yarn/pnpm
yarn install
# or
pnpm install
```

## Running the Application

### Step 1: Start Backend API

```bash
cd backend
source venv/bin/activate  # Activate virtual environment
python app.py
```

The API will be available at `http://localhost:5000`

### Step 2: Start Frontend Dev Server

In a new terminal:

```bash
npm run dev
```

The application will be available at `http://localhost:3000`

## Building for Production

### Frontend Build

```bash
npm run build
```

This creates optimized production build in `dist/` directory.

### Backend Deployment

For production, use a production WSGI server:

```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## API Endpoints

### `/health` (GET)
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda|cpu"
}
```

### `/transcribe` (POST)
Transcribe audio/video file

**Request:**
- `file`: Audio/video file (multipart/form-data)
- `language`: Language code (optional, default: "fa" for Farsi)

**Response:**
```json
{
  "status": "success",
  "filename": "audio.mp3",
  "language": "fa",
  "text": "Full transcription text...",
  "segments": [
    {
      "start": "00:00:00.000",
      "end": "00:00:05.500",
      "text": "سلام دنیا"
    }
  ]
}
```

### `/models` (GET)
Get available Whisper models

**Response:**
```json
{
  "available_models": ["tiny", "base", "small", "medium", "large"],
  "current_model": "medium",
  "description": "..."
}
```

### `/export` (POST)
Export transcription

**Request:**
```json
{
  "transcription": "Full text...",
  "segments": [...],
  "format": "txt|srt|vtt|json"
}
```

**Response:**
```json
{
  "status": "success",
  "format": "srt",
  "content": "...",
  "mime_type": "text/plain"
}
```

## Usage Guide

### 1. Add Files to Queue
- Click "Add Files" button in the left sidebar
- Select audio or video files
- Multiple files can be added to the queue

### 2. Transcribe
- Select a file from the queue
- Click "Transcribe" button
- Watch the progress indicator
- Results appear with timestamps

### 3. Search & Copy
- Use the search bar to find specific text
- Matching text is highlighted
- Click copy icon to copy individual segments

### 4. Export Results
- Select export format (TXT, SRT, VTT, JSON)
- Click "Export" button
- File is downloaded or ready to save

### 5. Theme Toggle
- Click sun/moon icon in header
- Switch between light and dark themes

## Project Structure

```
farsi_transcriber_web/
├── src/
│   ├── App.tsx              # Main application component
│   ├── main.tsx             # React entry point
│   ├── index.css            # Global styles
│   └── components/
│       ├── Button.tsx
│       ├── Progress.tsx
│       ├── Input.tsx
│       └── Select.tsx
├── backend/
│   ├── app.py               # Flask API server
│   ├── requirements.txt      # Python dependencies
│   └── .gitignore
├── public/
├── package.json
├── vite.config.ts
├── tsconfig.json
├── tailwind.config.js
├── postcss.config.js
└── README.md
```

## Configuration

### Environment Variables

Create a `.env.local` file in the root directory:

```
VITE_API_URL=http://localhost:5000
VITE_MAX_FILE_SIZE=500MB
```

### Backend Configuration

Edit `backend/app.py` to customize:

```python
# Change model size
model = whisper.load_model('large')  # tiny, base, small, medium, large

# Change upload folder
UPLOAD_FOLDER = '/custom/path'

# Change max file size
MAX_FILE_SIZE = 1024 * 1024 * 1024  # 1GB
```

## Troubleshooting

### Issue: "API connection failed"
**Solution**: Ensure backend is running on `http://localhost:5000`

### Issue: "Whisper model not found"
**Solution**: First run downloads the model (~3GB). Ensure internet connection and disk space.

### Issue: "CUDA out of memory"
**Solution**: Use smaller model or reduce batch size in `backend/app.py`

### Issue: "ffmpeg not found"
**Solution**: Install ffmpeg using your package manager (see Installation section)

### Issue: Port 3000 or 5000 already in use
**Solution**: Change ports in `vite.config.ts` and `backend/app.py`

## Performance Tips

1. **Use GPU** - Ensure NVIDIA CUDA is properly installed
2. **Choose appropriate model** - Balance speed vs accuracy
3. **Close other applications** - Free up RAM/VRAM
4. **Use SSD** - Faster model loading and file I/O
5. **Batch Processing** - Process multiple files sequentially

## Future Enhancements

- [ ] Drag-and-drop file upload
- [ ] Audio playback synchronized with transcription
- [ ] Edit segments inline
- [ ] Keyboard shortcuts
- [ ] Save/load sessions
- [ ] Speaker diarization
- [ ] Confidence scores
- [ ] Custom vocabulary support

## Development

### Code Style

```bash
# Format code (if ESLint configured)
npm run lint

# Build for development
npm run dev

# Build for production
npm run build
```

### Adding Components

New components go in `src/components/` and should:
- Use TypeScript
- Include prop interfaces
- Export as default
- Include JSDoc comments

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Models slow to load | GPU required for fast transcription |
| File not supported | Check file extension is in supported list |
| Transcription has errors | Try larger model (medium/large) |
| Application crashes | Check browser console and Flask logs |
| Export not working | Ensure segments data is complete |

## License

MIT License - Personal use and modifications allowed

## Credits

Built with:
- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [React](https://react.dev/) - UI framework
- [Vite](https://vitejs.dev/) - Build tool
- [Tailwind CSS](https://tailwindcss.com/) - Styling
- [Flask](https://flask.palletsprojects.com/) - Backend framework

## Support

For issues:
1. Check the troubleshooting section
2. Verify ffmpeg is installed
3. Check Flask backend logs
4. Review browser console for errors
5. Ensure Python 3.8+ and Node.js 16+ are installed
