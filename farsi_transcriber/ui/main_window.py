"""
Main application window for Farsi Transcriber

Provides PyQt6-based GUI for selecting files and transcribing Farsi audio/video.
"""

import os
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QTextEdit,
    QProgressBar,
    QFileDialog,
    QMessageBox,
)
from PyQt6.QtGui import QFont

from farsi_transcriber.models.whisper_transcriber import FarsiTranscriber
from farsi_transcriber.utils.export import TranscriptionExporter
from farsi_transcriber.ui.styles import get_stylesheet, get_color


class TranscriptionWorker(QThread):
    """Worker thread for transcription to prevent UI freezing"""

    # Signals
    progress_update = pyqtSignal(str)  # Status messages
    transcription_complete = pyqtSignal(dict)  # Results with timestamps
    error_occurred = pyqtSignal(str)  # Error messages

    def __init__(self, file_path: str, model_name: str = "medium"):
        super().__init__()
        self.file_path = file_path
        self.model_name = model_name
        self.transcriber = None

    def run(self):
        """Run transcription in background thread"""
        try:
            # Initialize Whisper transcriber
            self.progress_update.emit("Loading Whisper model...")
            self.transcriber = FarsiTranscriber(model_name=self.model_name)

            # Perform transcription
            self.progress_update.emit(f"Transcribing: {Path(self.file_path).name}")
            result = self.transcriber.transcribe(self.file_path)

            # Format result for display with timestamps
            display_text = self.transcriber.format_result_for_display(result)

            # Add full text for export
            result["full_text"] = result.get("text", "")

            self.progress_update.emit("Transcription complete!")
            self.transcription_complete.emit(
                {
                    "text": display_text,
                    "segments": result.get("segments", []),
                    "full_text": result.get("text", ""),
                }
            )

        except Exception as e:
            self.error_occurred.emit(f"Error: {str(e)}")


class MainWindow(QMainWindow):
    """Main application window for Farsi Transcriber"""

    # Supported audio and video formats
    SUPPORTED_FORMATS = (
        "Audio Files (*.mp3 *.wav *.m4a *.flac *.ogg *.aac *.wma);;",
        "Video Files (*.mp4 *.mkv *.mov *.webm *.avi *.flv *.wmv);;",
        "All Files (*.*)",
    )

    def __init__(self):
        super().__init__()
        self.selected_file = None
        self.transcription_worker = None
        self.last_result = None
        # Apply stylesheet
        self.setStyleSheet(get_stylesheet())
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Farsi Transcriber")
        self.setGeometry(100, 100, 900, 700)

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title_label = QLabel("Farsi Audio/Video Transcriber")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        main_layout.addWidget(title_label)

        # File selection section
        file_section_layout = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("color: gray;")
        file_section_layout.addWidget(self.file_label, 1)

        self.select_button = QPushButton("Select File")
        self.select_button.clicked.connect(self.on_select_file)
        file_section_layout.addWidget(self.select_button)

        main_layout.addLayout(file_section_layout)

        # Transcribe button
        self.transcribe_button = QPushButton("Transcribe")
        self.transcribe_button.clicked.connect(self.on_transcribe)
        self.transcribe_button.setEnabled(False)
        main_layout.addWidget(self.transcribe_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #666; font-style: italic;")
        main_layout.addWidget(self.status_label)

        # Results text area
        results_title = QLabel("Transcription Results:")
        results_title_font = QFont()
        results_title_font.setBold(True)
        results_title.setFont(results_title_font)
        main_layout.addWidget(results_title)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setPlaceholderText(
            "Transcription results will appear here..."
        )
        # Set monospace font for results
        mono_font = QFont("Courier New", 10)
        self.results_text.setFont(mono_font)
        main_layout.addWidget(self.results_text)

        # Buttons layout (Export, Clear)
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()

        self.export_button = QPushButton("Export Results")
        self.export_button.clicked.connect(self.on_export)
        self.export_button.setEnabled(False)
        buttons_layout.addWidget(self.export_button)

        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.on_clear)
        buttons_layout.addWidget(self.clear_button)

        main_layout.addLayout(buttons_layout)

    def on_select_file(self):
        """Handle file selection dialog"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio or Video File", "", "".join(self.SUPPORTED_FORMATS)
        )

        if file_path:
            self.selected_file = file_path
            file_name = Path(file_path).name
            self.file_label.setText(f"Selected: {file_name}")
            self.file_label.setStyleSheet("color: #333;")
            self.transcribe_button.setEnabled(True)
            self.export_button.setEnabled(False)
            self.results_text.clear()
            self.status_label.setText("File selected. Click 'Transcribe' to start.")

    def on_transcribe(self):
        """Handle transcription button click"""
        if not self.selected_file:
            QMessageBox.warning(self, "Error", "Please select a file first.")
            return

        # Disable buttons during transcription
        self.transcribe_button.setEnabled(False)
        self.select_button.setEnabled(False)
        self.export_button.setEnabled(False)

        # Show progress
        self.progress_bar.setVisible(True)
        self.status_label.setText("Transcribing...")

        # Create and start worker thread
        self.transcription_worker = TranscriptionWorker(self.selected_file)
        self.transcription_worker.progress_update.connect(self.on_progress_update)
        self.transcription_worker.transcription_complete.connect(
            self.on_transcription_complete
        )
        self.transcription_worker.error_occurred.connect(self.on_error)
        self.transcription_worker.start()

    def on_progress_update(self, message: str):
        """Handle progress updates from worker thread"""
        self.status_label.setText(message)

    def on_transcription_complete(self, result: dict):
        """Handle completed transcription"""
        self.progress_bar.setVisible(False)
        self.transcribe_button.setEnabled(True)
        self.select_button.setEnabled(True)
        self.export_button.setEnabled(True)
        self.status_label.setText("Transcription complete!")

        # Display results with timestamps
        self.results_text.setText(result.get("text", "No transcription available"))

        # Store result for export
        self.last_result = result

    def on_error(self, error_message: str):
        """Handle errors from worker thread"""
        self.progress_bar.setVisible(False)
        self.transcribe_button.setEnabled(True)
        self.select_button.setEnabled(True)
        self.status_label.setText("Error occurred. Check message below.")
        QMessageBox.critical(self, "Transcription Error", error_message)

    def on_export(self):
        """Handle export button click"""
        if not self.last_result:
            QMessageBox.warning(self, "Warning", "No transcription to export.")
            return

        file_path, file_filter = QFileDialog.getSaveFileName(
            self,
            "Export Transcription",
            "",
            "Text Files (*.txt);;SRT Subtitles (*.srt);;WebVTT Subtitles (*.vtt);;JSON (*.json);;TSV (*.tsv)",
        )

        if file_path:
            try:
                file_path = Path(file_path)

                # Determine format from file extension
                suffix = file_path.suffix.lower().lstrip(".")
                if not suffix:
                    # Default to txt if no extension
                    suffix = "txt"
                    file_path = file_path.with_suffix(".txt")

                # Export using the appropriate format
                TranscriptionExporter.export(self.last_result, file_path, suffix)

                QMessageBox.information(
                    self,
                    "Success",
                    f"Transcription exported successfully to:\n{file_path.name}",
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Export Error", f"Failed to export: {str(e)}"
                )

    def on_clear(self):
        """Clear all results and reset UI"""
        self.selected_file = None
        self.file_label.setText("No file selected")
        self.file_label.setStyleSheet("color: gray;")
        self.results_text.clear()
        self.status_label.setText("Ready")
        self.transcribe_button.setEnabled(False)
        self.export_button.setEnabled(False)
