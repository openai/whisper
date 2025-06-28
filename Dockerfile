# Dockerfile para Whisper + Streamlit + FastAPI
FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Copiar archivos del proyecto
COPY . /app

# Instalar dependencias de Python
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir streamlit fastapi uvicorn

# Puerto para Streamlit y FastAPI
EXPOSE 8501 8000

# Comando por defecto: arranca ambos servicios
CMD streamlit run app.py & uvicorn api:app --host 0.0.0.0 --port 8000
