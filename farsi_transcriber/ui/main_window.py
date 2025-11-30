"""
Main application window for Farsi Transcriber

Provides PyQt6-based GUI for selecting files and transcribing Farsi audio/video.
"""

import os
from pathlib import Path
from typing import List

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
    QListWidget,
    QListWidgetItem,
    QSplitter
)
from PyQt6.QtGui import QFont, QColor, QIcon

from farsi_transcriber.core.transcriber import FarsiTranscriber
from farsi_transcriber.core.export import TranscriptionExporter
from farsi_transcriber.ui.styles import get_stylesheet, get_color


class TranscriptionWorker(QThread):
    """Worker thread for transcription to prevent UI freezing"""

    # Signals
    progress_update = pyqtSignal(str)  # Status messages
    item_complete = pyqtSignal(str, dict) # file_path, result
    item_error = pyqtSignal(str, str) # file_path, error_message
    queue_complete = pyqtSignal()

    def __init__(self, file_queue: List[str], model_name: str = "medium"):
        super().__init__()
        self.file_queue = file_queue
        self.model_name = model_name
        self.is_running = True

    def run(self):
        """Run transcription in background thread"""
        try:
            self.progress_update.emit("Loading Whisper model...")
            transcriber = FarsiTranscriber(model_name=self.model_name)

            for file_path in self.file_queue:
                if not self.is_running:
                    break

                try:
                    self.progress_update.emit(f"Transcribing: {Path(file_path).name}")
                    result = transcriber.transcribe(file_path)

                    # Add full text for export
                    result["full_text"] = result.get("text", "")

                    self.item_complete.emit(file_path, result)

                except Exception as e:
                    self.item_error.emit(file_path, str(e))

            self.queue_complete.emit()

        except Exception as e:
            self.progress_update.emit(f"Critical Error: {str(e)}")

    def stop(self):
        self.is_running = False


class MainWindow(QMainWindow):
    """Main application window for Farsi Transcriber"""

    # Supported audio and video formats
    SUPPORTED_FORMATS = (
        "Media Files (*.mp3 *.wav *.m4a *.flac *.ogg *.aac *.wma *.mp4 *.mkv *.mov *.webm *.avi *.flv *.wmv);;",
        "All Files (*.*)",
    )

    def __init__(self):
        super().__init__()
        self.file_queue = [] # List of file paths
        self.results = {} # Map file_path -> result dict
        self.transcription_worker = None

        # Apply stylesheet
        self.setStyleSheet(get_stylesheet())
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Farsi Transcriber")
        self.setGeometry(100, 100, 1000, 700)

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

        # Splitter for Queue and Results
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter, 1)

        # LEFT SIDE: File Queue
        queue_widget = QWidget()
        queue_layout = QVBoxLayout(queue_widget)
        queue_layout.setContentsMargins(0, 0, 0, 0)

        queue_label = QLabel("File Queue")
        queue_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        queue_layout.addWidget(queue_label)

        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.on_file_selected)
        queue_layout.addWidget(self.file_list)

        # Queue Buttons
        queue_btn_layout = QHBoxLayout()
        self.add_button = QPushButton("Add Files")
        self.add_button.clicked.connect(self.on_add_files)
        queue_btn_layout.addWidget(self.add_button)

        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self.on_remove_file)
        queue_btn_layout.addWidget(self.remove_button)

        queue_layout.addLayout(queue_btn_layout)
        splitter.addWidget(queue_widget)

        # RIGHT SIDE: Results
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        results_layout.setContentsMargins(10, 0, 0, 0)

        results_title = QLabel("Transcription Results:")
        results_title.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        results_layout.addWidget(results_title)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setPlaceholderText(
            "Select a processed file to view results..."
        )
        mono_font = QFont("Courier New", 10)
        self.results_text.setFont(mono_font)
        results_layout.addWidget(self.results_text)

        splitter.addWidget(results_widget)
        splitter.setSizes([300, 700])

        # Bottom Controls
        bottom_layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        bottom_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #666; font-style: italic;")
        bottom_layout.addWidget(self.status_label)

        action_layout = QHBoxLayout()

        self.transcribe_button = QPushButton("Start Transcription")
        self.transcribe_button.clicked.connect(self.on_transcribe)
        self.transcribe_button.setEnabled(False)
        # Style it prominent
        self.transcribe_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px;
            }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        action_layout.addWidget(self.transcribe_button)

        action_layout.addStretch()

        self.export_button = QPushButton("Export Current")
        self.export_button.clicked.connect(self.on_export)
        self.export_button.setEnabled(False)
        action_layout.addWidget(self.export_button)

        self.clear_button = QPushButton("Clear All")
        self.clear_button.clicked.connect(self.on_clear)
        action_layout.addWidget(self.clear_button)

        bottom_layout.addLayout(action_layout)
        main_layout.addLayout(bottom_layout)

    def on_add_files(self):
        """Handle file addition"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Audio or Video Files", "", "".join(self.SUPPORTED_FORMATS)
        )

        if file_paths:
            for path in file_paths:
                if path not in self.file_queue:
                    self.file_queue.append(path)
                    item = QListWidgetItem(Path(path).name)
                    item.setData(Qt.ItemDataRole.UserRole, path)
                    # Set icon (pending)
                    item.setForeground(QColor("black"))
                    self.file_list.addItem(item)

            self.transcribe_button.setEnabled(len(self.file_queue) > 0)
            self.status_label.setText(f"{len(self.file_queue)} files in queue.")

    def on_remove_file(self):
        row = self.file_list.currentRow()
        if row >= 0:
            item = self.file_list.takeItem(row)
            path = item.data(Qt.ItemDataRole.UserRole)
            if path in self.file_queue:
                self.file_queue.remove(path)
            if path in self.results:
                del self.results[path]

            self.transcribe_button.setEnabled(len(self.file_queue) > 0)

    def on_file_selected(self, item):
        path = item.data(Qt.ItemDataRole.UserRole)
        if path in self.results:
            result = self.results[path]
            # Format nicely
            text = self._format_result_display(result)
            self.results_text.setText(text)
            self.export_button.setEnabled(True)
        else:
            self.results_text.clear()
            self.results_text.setPlaceholderText("No results yet for this file.")
            self.export_button.setEnabled(False)

    def on_transcribe(self):
        """Handle transcription button click"""
        if not self.file_queue:
            return

        pending_files = [f for f in self.file_queue if f not in self.results]
        if not pending_files:
             QMessageBox.information(self, "Info", "All files in queue are already transcribed.")
             return

        # Disable input
        self.transcribe_button.setEnabled(False)
        self.add_button.setEnabled(False)
        self.remove_button.setEnabled(False)
        self.clear_button.setEnabled(False)

        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0) # Indeterminate
        self.status_label.setText("Starting batch transcription...")

        # Create and start worker
        self.transcription_worker = TranscriptionWorker(pending_files)
        self.transcription_worker.progress_update.connect(self.on_progress_update)
        self.transcription_worker.item_complete.connect(self.on_item_complete)
        self.transcription_worker.item_error.connect(self.on_item_error)
        self.transcription_worker.queue_complete.connect(self.on_queue_complete)
        self.transcription_worker.start()

    def on_progress_update(self, message: str):
        self.status_label.setText(message)

    def on_item_complete(self, file_path, result):
        self.results[file_path] = result

        # Find item in list and mark green
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == file_path:
                item.setForeground(QColor("green"))
                item.setText(f"✓ {Path(file_path).name}")
                break

    def on_item_error(self, file_path, error):
        # Find item in list and mark red
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == file_path:
                item.setForeground(QColor("red"))
                item.setText(f"✗ {Path(file_path).name}")
                item.setToolTip(error)
                break

    def on_queue_complete(self):
        self.progress_bar.setVisible(False)
        self.transcribe_button.setEnabled(True)
        self.add_button.setEnabled(True)
        self.remove_button.setEnabled(True)
        self.clear_button.setEnabled(True)
        self.status_label.setText("Batch transcription complete!")
        QMessageBox.information(self, "Done", "All files have been processed.")

    def on_export(self):
        row = self.file_list.currentRow()
        if row < 0:
            return

        item = self.file_list.item(row)
        path = item.data(Qt.ItemDataRole.UserRole)

        if path not in self.results:
            QMessageBox.warning(self, "Warning", "This file has not been transcribed yet.")
            return

        result = self.results[path]

        file_path, file_filter = QFileDialog.getSaveFileName(
            self,
            f"Export {Path(path).stem}",
            "",
            "Text Files (*.txt);;SRT Subtitles (*.srt);;WebVTT Subtitles (*.vtt);;JSON (*.json);;TSV (*.tsv)",
        )

        if file_path:
            try:
                file_path = Path(file_path)
                suffix = file_path.suffix.lower().lstrip(".")
                if not suffix:
                    suffix = "txt"
                    file_path = file_path.with_suffix(".txt")

                TranscriptionExporter.export(result, file_path, suffix)
                QMessageBox.information(self, "Success", f"Exported to {file_path.name}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))

    def on_clear(self):
        self.file_queue = []
        self.results = []
        self.file_list.clear()
        self.results_text.clear()
        self.status_label.setText("Ready")
        self.transcribe_button.setEnabled(False)

    def _format_result_display(self, result):
        # Quick helper to format text for display
        lines = []
        for segment in result.get("segments", []):
            start = self._format_time(segment.get("start", 0))
            end = self._format_time(segment.get("end", 0))
            text = segment.get("text", "").strip()
            lines.append(f"[{start} - {end}]\n{text}\n")
        return "\n".join(lines)

    @staticmethod
    def _format_time(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
