import streamlit as st
import torch
import numpy as np
import os
from whisper.transcribe import transcribe
from whisper import load_model
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE

st.set_page_config(page_title="Whisper Transcriber", layout="wide")
st.title("Whisper Transcriber - Interfaz Streamlit")

# Sidebar options
st.sidebar.header("Opciones de transcripción")

# Idiomas para el selector (nombre completo)
language_names = ["(autodetect)"] + [LANGUAGES[k].title() for k in sorted(LANGUAGES.keys())]
language_key_map = dict(zip([LANGUAGES[k].title() for k in sorted(LANGUAGES.keys())], sorted(LANGUAGES.keys())))

# Modelo
model_name = st.sidebar.selectbox(
    "Modelo",
    ["tiny", "base", "small", "medium", "large", "turbo"],
    index=5,
    help="Selecciona el modelo Whisper a utilizar. Modelos más grandes suelen ser más precisos pero más lentos."
)

# Tarea
task = st.sidebar.selectbox(
    "Tarea",
    ["transcribe", "translate"],
    index=0,
    help="'transcribe' transcribe el audio en el mismo idioma. 'translate' traduce el audio al inglés."
)

# Idioma
language_display = st.sidebar.selectbox(
    "Idioma (deja vacío para autodetectar)",
    language_names,
    index=0,
    help="Selecciona el idioma hablado en el audio. Si eliges '(autodetect)', el modelo intentará detectarlo automáticamente."
)
if language_display == "(autodetect)":
    language = None
else:
    language = language_key_map[language_display]

# Temperatura
st.sidebar.markdown("<span style='font-size: 12px; color: gray;'>La temperatura controla la aleatoriedad de la transcripción. Valores bajos (0) hacen la salida más determinista, valores altos pueden generar resultados más creativos pero menos precisos.</span>", unsafe_allow_html=True)
temperature = st.sidebar.slider(
    "Temperatura",
    0.0, 1.0, 0.0, 0.1,
    help="Controla la aleatoriedad de la transcripción. 0 es determinista, valores más altos pueden dar resultados más variados."
)

# Word timestamps
word_timestamps = st.sidebar.checkbox(
    "Timestamps por palabra",
    value=False,
    help="Si está activado, se mostrarán marcas de tiempo para cada palabra."
)

# Selector de destino de salida
output_mode = st.sidebar.radio(
    "¿Dónde quieres guardar la salida?",
    ("En mi equipo (local)", "En la ruta del contenedor (/app/output)"),
    help="Elige si quieres guardar los archivos en tu equipo local o en la carpeta compartida del contenedor."
)

if output_mode == "En mi equipo (local)":
    st.sidebar.markdown("<small>Selecciona cualquier archivo de la carpeta donde quieras guardar la salida. Se usará la ruta de esa carpeta.</small>", unsafe_allow_html=True)
    selected_file = st.sidebar.file_uploader(
        "Selecciona un archivo de la carpeta destino (solo para elegir la carpeta)",
        type=None,
        accept_multiple_files=False,
        key="folder_selector"
    )
    if selected_file is not None and hasattr(selected_file, 'name'):
        import pathlib
        # En local, file_uploader solo da el nombre, no la ruta absoluta, así que usamos cwd
        selected_folder = os.getcwd()
    else:
        selected_folder = os.getcwd()
    default_dir = selected_folder
else:
    default_dir = "/app/output"

output_dir = st.sidebar.text_input(
    "Carpeta de salida para los resultados",
    value=default_dir,
    help="Ruta donde se guardarán los archivos de salida generados por la transcripción. Puedes personalizarla si lo deseas."
)

# Archivo de audio
st.header("Selecciona archivos de audio para transcribir")
audio_files = st.file_uploader(
    "Selecciona uno o varios archivos de audio",
    type=["wav", "mp3", "m4a", "flac", "ogg"],
    accept_multiple_files=True
)

if st.button("Transcribir") and audio_files:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.info(f"Cargando modelo '{model_name}' en {device}...")
    model = load_model(model_name, device=device)
    os.makedirs(output_dir, exist_ok=True)
    for idx, audio_file in enumerate(audio_files):
        st.write(f"Procesando archivo: {audio_file.name}")
        with st.spinner(f"Transcribiendo {audio_file.name}..."):
            # Guardar archivo temporalmente
            temp_path = f"temp_{audio_file.name}"
            with open(temp_path, "wb") as f:
                f.write(audio_file.read())
            # Barra de progreso
            progress = st.progress(0)
            def progress_callback(p):
                progress.progress(int(p * 100))
            # Ejecutar transcripción
            result = transcribe(
                model,
                temp_path,
                task=task,
                language=language,
                temperature=temperature,
                word_timestamps=word_timestamps,
                verbose=True
            )
            # Guardar resultado en archivo de texto en la carpeta de salida
            output_txt = os.path.join(output_dir, f"{os.path.splitext(audio_file.name)[0]}.txt")
            with open(output_txt, "w", encoding="utf-8") as f:
                f.write(result["text"])
            # Mostrar resultado
            st.subheader(f"Transcripción de {audio_file.name}")
            st.text_area("Texto transcrito", result["text"], height=200)
            st.write("Detalles:", result)
            # Botón de descarga siempre disponible
            st.download_button(
                label="Descargar transcripción",
                data=result["text"],
                file_name=f"{os.path.splitext(audio_file.name)[0]}.txt",
                mime="text/plain"
            )
            os.remove(temp_path)
            progress.progress(100)
    st.success("¡Transcripción completada!")
else:
    st.info("Selecciona archivos y pulsa 'Transcribir'.")
