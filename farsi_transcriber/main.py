#!/usr/bin/env python3
"""
Farsi Transcriber - Main Entry Point

A PyQt6-based desktop application for transcribing Farsi audio and video files.
"""

import sys
from PyQt6.QtWidgets import QApplication

from farsi_transcriber.ui.main_window import MainWindow


def main():
    """Main entry point for the application"""
    app = QApplication(sys.argv)

    # Create and show main window
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
