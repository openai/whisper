from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import os
import shutil
from whisper.transcribe import transcribe
from whisper import load_model
from whisper.tokenizer import LANGUAGES
import torch

app = FastAPI()

# Cargar modelo solo una vez al iniciar el contenedor
MODEL_NAME = os.environ.get("WHISPER_MODEL", "turbo")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(MODEL_NAME, device=DEVICE)

@app.post("/transcribe")
def transcribe_audio(
    file: UploadFile = File(...),
    task: str = Form("transcribe"),
    language: str = Form(None),
    temperature: float = Form(0.0),
    word_timestamps: bool = Form(False)
):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        result = transcribe(
            model,
            temp_path,
            task=task,
            language=language,
            temperature=temperature,
            word_timestamps=word_timestamps,
            verbose=False
        )
        os.remove(temp_path)
        return JSONResponse({
            "text": result["text"],
            "segments": result.get("segments", []),
            "language": result.get("language", "")
        })
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return JSONResponse({"error": str(e)}, status_code=500)
