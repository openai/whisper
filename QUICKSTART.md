# Farsi Transcriber - Quick Start Guide

You now have **TWO** complete applications for Farsi transcription:

## 🖥️ Option 1: Desktop App (PyQt6)

**Location:** `/home/user/whisper/farsi_transcriber/`

### Setup
```bash
cd farsi_transcriber
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

**Features:**
- ✅ Standalone desktop application
- ✅ Works completely offline
- ✅ Direct access to file system
- ✅ Lightweight and fast
- ⚠️ Simpler UI (green theme)

**Good for:**
- Local-only transcription
- Users who prefer desktop apps
- Offline processing

---

## 🌐 Option 2: Web App (React + Flask)

**Location:** `/home/user/whisper/farsi_transcriber_web/`

### Setup

**Backend (Flask):**
```bash
cd farsi_transcriber_web/backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
# API runs on http://localhost:5000
```

**Frontend (React):**
```bash
cd farsi_transcriber_web
npm install
npm run dev
# App runs on http://localhost:3000
```

**Features:**
- ✅ Modern web-based UI (matches your Figma design exactly)
- ✅ File queue management
- ✅ Dark/Light theme toggle
- ✅ Search with text highlighting
- ✅ Copy segments to clipboard
- ✅ Resizable window
- ✅ RTL support for Farsi
- ✅ Multiple export formats
- ✅ Professional styling

**Good for:**
- Modern web experience
- Team collaboration (can be deployed online)
- More features and polish
- Professional appearance

---

## 📊 Comparison

| Feature | Desktop (PyQt6) | Web (React) |
|---------|-----------------|------------|
| **Interface** | Simple, green | Modern, professional |
| **Dark Mode** | ❌ | ✅ |
| **File Queue** | ❌ | ✅ |
| **Search** | ❌ | ✅ |
| **Copy Segments** | ❌ | ✅ |
| **Resizable Window** | ❌ | ✅ |
| **Export Formats** | SRT, TXT, VTT, JSON, TSV | TXT, SRT, VTT, JSON |
| **Offline** | ✅ | Requires backend |
| **Easy Setup** | ✅✅ | ✅ (2 terminals) |
| **Deployment** | Desktop only | Can host online |
| **Code Size** | ~25KB | ~200KB |

---

## 🚀 Which Should You Use?

### Use **Desktop App** if:
- You want simple, quick setup
- You never share transcriptions
- You prefer offline processing
- You don't need advanced features

### Use **Web App** if:
- You like modern interfaces
- You want dark/light themes
- You need file queue management
- You want to potentially share online
- You want professional appearance

---

## 📁 Project Structure

```
whisper/
├── farsi_transcriber/              (Desktop PyQt6 App)
│   ├── ui/
│   ├── models/
│   ├── utils/
│   ├── config.py
│   ├── main.py
│   └── requirements.txt
│
└── farsi_transcriber_web/          (Web React App)
    ├── src/
    │   ├── App.tsx
    │   ├── components/
    │   └── main.tsx
    ├── backend/
    │   ├── app.py
    │   └── requirements.txt
    ├── package.json
    └── vite.config.ts
```

---

## 🔧 System Requirements

### Desktop App
- Python 3.8+
- ffmpeg
- 4GB RAM

### Web App
- Python 3.8+ (backend)
- Node.js 16+ (frontend)
- ffmpeg
- 4GB RAM

---

## 📝 Setup Checklist

### Initial Setup (One-time)

- [ ] Install ffmpeg
  ```bash
  # Ubuntu/Debian
  sudo apt install ffmpeg

  # macOS
  brew install ffmpeg

  # Windows
  choco install ffmpeg
  ```

- [ ] Verify Python 3.8+
  ```bash
  python3 --version
  ```

- [ ] Verify Node.js 16+ (for web app only)
  ```bash
  node --version
  ```

### Desktop App Setup

- [ ] Create virtual environment
- [ ] Install requirements
- [ ] Run app

### Web App Setup

**Backend:**
- [ ] Create virtual environment
- [ ] Install requirements
- [ ] Run Flask server

**Frontend:**
- [ ] Install Node dependencies
- [ ] Run dev server

---

## 🎯 Quick Start (Fastest)

### Desktop (30 seconds)
```bash
cd whisper/farsi_transcriber
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt && python main.py
```

### Web (2 minutes)
Terminal 1:
```bash
cd whisper/farsi_transcriber_web/backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt && python app.py
```

Terminal 2:
```bash
cd whisper/farsi_transcriber_web
npm install && npm run dev
```

---

## 🐛 Troubleshooting

### "ffmpeg not found"
Install ffmpeg (see requirements above)

### "ModuleNotFoundError" (Python)
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### "npm: command not found"
Install Node.js from https://nodejs.org

### App runs slow
- Use GPU: Install CUDA
- Reduce model size: change to 'small' or 'tiny'
- Close other applications

---

## 📚 Full Documentation

- **Desktop App:** `farsi_transcriber/README.md`
- **Web App:** `farsi_transcriber_web/README.md`
- **API Docs:** `farsi_transcriber_web/README.md` (Endpoints section)

---

## 🎓 What Was Built

### Desktop Application (PyQt6)
✅ File picker for audio/video
✅ Whisper integration with word-level timestamps
✅ 5 export formats (TXT, SRT, VTT, JSON, TSV)
✅ Professional styling
✅ Progress indicators
✅ Threading to prevent UI freezing

### Web Application (React + Flask)
✅ Complete Figma design implementation
✅ File queue management
✅ Dark/light theme
✅ Search with highlighting
✅ Segment management
✅ Resizable window
✅ RTL support
✅ Flask backend with Whisper integration
✅ 4 export formats
✅ Real file upload handling

---

## 🚀 Next Steps

1. **Choose your app** (Desktop or Web)
2. **Install ffmpeg** if not already installed
3. **Follow the setup instructions** above
4. **Test with a Farsi audio file**
5. **Export in your preferred format**

---

## 💡 Tips

- **First transcription is slow** (downloads 769MB model)
- **Use larger models** (medium/large) for better accuracy
- **Use smaller models** (tiny/base) for speed
- **GPU significantly speeds up** transcription
- **Both apps work offline** (after initial model download)

---

## 📧 Need Help?

- Check the full README in each app's directory
- Verify all requirements are installed
- Check browser console (web app) or Python output (desktop)
- Ensure ffmpeg is in your PATH

---

**Enjoy your Farsi transcription apps!** 🎉
