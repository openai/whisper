#!/usr/bin/env python3
"""
Voice Recording and Transcription Script
Records audio from microphone and converts it to text using OpenAI Whisper
"""

import sounddevice as sd
import numpy as np
import whisper
import tempfile
import wave
import os
import re
import pyperclip
import sys
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import json
import warnings
from datetime import datetime
from pathlib import Path
from pynput import keyboard
import pystray
from PIL import Image, ImageDraw

# Suppress common PyTorch/Whisper warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")
warnings.filterwarnings("ignore", message=".*FP16.*")
warnings.filterwarnings("ignore", category=UserWarning, module="whisper")

# Set environment variable to reduce PyTorch verbosity
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class PromptProcessor:
    """Processes transcribed text to create better Claude Code prompts"""

    def __init__(self):
        self.patterns = [
            # Agent references
            (r'\buse agent ([\w-]+)\b', r'@agent \1'),
            (r'\blaunch agent ([\w-]+(?:\s+[\w-]+)*)\b', lambda m: f"@agent {m.group(1).replace(' ', '-')}"),
            (r'\bcall agent ([\w-]+(?:\s+[\w-]+)*)\b', lambda m: f"@agent {m.group(1).replace(' ', '-')}"),

            # Tool references
            (r'\brun tool (\w+)\b', r'@tool \1'),
            (r'\bcall the (\w+) tool\b', r'@tool \1'),
            (r'\buse the (\w+) tool\b', r'@tool \1'),

            # Directory references
            (r'\bdirectory ([\w/\\.-]+)\b', r'@dir \1/'),
            (r'\bfolder ([\w/\\.-]+)\b', r'@dir \1/'),
            (r'\bthe ([\w.-]+) directory\b', r'@dir \1/'),

            # File references
            (r'\bfile ([\w/\\.-]+\.[\w]+)\b', r'@file \1'),
            (r'\bthe ([\w.-]+\.[\w]+) file\b', r'@file \1'),
            (r'\breadme file\b', '@file README.md'),
            (r'\bpackage json\b', '@file package.json'),

            # Code elements
            (r'\bfunction ([\w_]+)\b', r'`\1()` function'),
            (r'\bclass ([\w_]+)\b', r'`\1` class'),
            (r'\bvariable ([\w_]+)\b', r'`\1` variable'),
            (r'\bmethod ([\w_]+)\b', r'`\1()` method'),

            # Task management
            (r'\badd to todo\b', 'add to todo:'),
            (r'\bnew task\b', 'new todo:'),
            (r'\bmark complete\b', 'mark todo complete'),
            (r'\bmark done\b', 'mark todo complete'),

            # Commands
            (r'\brun tests\b', 'run tests'),
            (r'\bcommit changes\b', 'commit changes'),
            (r'\bcreate pull request\b', 'create PR'),
            (r'\binstall dependencies\b', 'install dependencies'),
        ]

    def process(self, text):
        """Process raw transcription into a Claude Code prompt"""
        processed = text.strip()

        # Apply pattern replacements
        for pattern, replacement in self.patterns:
            if callable(replacement):
                processed = re.sub(pattern, replacement, processed, flags=re.IGNORECASE)
            else:
                processed = re.sub(pattern, replacement, processed, flags=re.IGNORECASE)

        # Capitalize first letter and ensure proper punctuation
        if processed:
            processed = processed[0].upper() + processed[1:] if len(processed) > 1 else processed.upper()
            if not processed.endswith(('.', '!', '?', ':')):
                processed += '.'

        return processed

class SettingsManager:
    """Manages application settings with JSON persistence"""

    def __init__(self):
        self.settings_file = Path('voice_to_text_settings.json')
        self.default_settings = {
            'hotkey': 'f1',
            'always_on_top': False,
            'minimize_to_tray': True,
            'whisper_model': 'base',
            'window_geometry': '600x500',
            'auto_copy_clipboard': True
        }
        self.settings = self.load_settings()

    def load_settings(self):
        """Load settings from JSON file or create defaults"""
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                    # Merge with defaults to handle new settings
                    merged = self.default_settings.copy()
                    merged.update(settings)
                    return merged
        except Exception as e:
            print(f"Error loading settings: {e}")
        return self.default_settings.copy()

    def save_settings(self):
        """Save current settings to JSON file"""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=2)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def get(self, key, default=None):
        """Get a setting value"""
        return self.settings.get(key, default)

    def set(self, key, value):
        """Set a setting value and save"""
        self.settings[key] = value
        self.save_settings()

class VoiceRecorder:
    def __init__(self, sample_rate=16000, channels=1, settings_manager=None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.recording = False
        self.audio_data = []
        self.processor = PromptProcessor()
        self.settings = settings_manager or SettingsManager()

        # Ensure transcripts directory exists
        self.transcripts_dir = Path('transcripts')
        self.transcripts_dir.mkdir(exist_ok=True)

    def record_audio(self, duration=None):
        """
        Record audio from microphone
        Args:
            duration: Recording duration in seconds. If None, records until Enter is pressed
        """
        print("Loading Whisper model...")
        model = whisper.load_model("base")

        if duration:
            print(f"Recording for {duration} seconds...")
            audio = sd.rec(int(duration * self.sample_rate),
                          samplerate=self.sample_rate,
                          channels=self.channels,
                          dtype=np.float32)
            sd.wait()
            print("Recording complete!")
        else:
            print("Recording... Press Enter to stop.")
            self.recording = True
            self.audio_data = []

            def callback(indata, frames, time, status):
                if self.recording:
                    self.audio_data.append(indata.copy())

            with sd.InputStream(callback=callback,
                              samplerate=self.sample_rate,
                              channels=self.channels,
                              dtype=np.float32):
                input()  # Wait for Enter key
                self.recording = False

            if self.audio_data:
                audio = np.concatenate(self.audio_data, axis=0)
            else:
                print("No audio recorded.")
                return

        # Save to temporary file
        temp_file = self._save_to_temp_file(audio)

        try:
            # Transcribe with Whisper
            print("Transcribing...")
            result = model.transcribe(temp_file)

            # Process the transcription
            raw_text = result["text"]
            processed_text = self.processor.process(raw_text)

            # Display results
            print("\n" + "="*50)
            print("RAW TRANSCRIPTION:")
            print("="*50)
            print(raw_text)
            print("\n" + "="*50)
            print("PROCESSED PROMPT:")
            print("="*50)
            print(processed_text)
            print("="*50)

            # Copy processed text to clipboard
            try:
                pyperclip.copy(processed_text)
                print("\n✓ Processed prompt copied to clipboard!")
                print("You can now paste it directly into Claude Code.")
            except Exception as e:
                print(f"\n⚠ Could not copy to clipboard: {e}")
                print("Please copy the processed text manually.")

            # Save to file in transcripts directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.transcripts_dir / f"transcription_{timestamp}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Transcription - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*60 + "\n")
                f.write("RAW TRANSCRIPTION:\n")
                f.write("="*60 + "\n")
                f.write(raw_text + "\n\n")
                f.write("="*60 + "\n")
                f.write("PROCESSED PROMPT:\n")
                f.write("="*60 + "\n")
                f.write(processed_text)

            print(f"\nTranscription saved to: {output_file}")

        finally:
            # Clean up temporary file
            os.unlink(temp_file)

    def _save_to_temp_file(self, audio_data):
        """Save audio data to temporary WAV file"""
        temp_file = tempfile.mktemp(suffix=".wav")

        with wave.open(temp_file, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)

            # Convert float32 to int16
            audio_int16 = (audio_data * 32767).astype(np.int16)
            wf.writeframes(audio_int16.tobytes())

        return temp_file

class SettingsDialog:
    """Settings dialog for configuring the voice recorder"""

    def __init__(self, parent, settings_manager, apply_callback):
        self.settings = settings_manager
        self.apply_callback = apply_callback

        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Settings")
        self.dialog.geometry("400x300")
        self.dialog.resizable(False, False)
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # Center the dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (400 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (300 // 2)
        self.dialog.geometry(f"400x300+{x}+{y}")

        self.create_widgets()

    def create_widgets(self):
        """Create the settings widgets"""
        main_frame = ttk.Frame(self.dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Hotkey setting
        ttk.Label(main_frame, text="Global Hotkey:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        hotkey_frame = ttk.Frame(main_frame)
        hotkey_frame.pack(fill=tk.X, pady=(0, 15))

        self.hotkey_var = tk.StringVar(value=self.settings.get('hotkey', 'f1'))
        hotkey_combo = ttk.Combobox(hotkey_frame, textvariable=self.hotkey_var,
                                   values=['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12'],
                                   state="readonly", width=10)
        hotkey_combo.pack(side=tk.LEFT)
        ttk.Label(hotkey_frame, text="(Press this key anywhere to start/stop recording)").pack(side=tk.LEFT, padx=(10, 0))

        # Whisper model setting
        ttk.Label(main_frame, text="Whisper Model:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        model_frame = ttk.Frame(main_frame)
        model_frame.pack(fill=tk.X, pady=(0, 15))

        self.model_var = tk.StringVar(value=self.settings.get('whisper_model', 'base'))
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_var,
                                  values=['tiny', 'base', 'small', 'medium', 'large', 'turbo'],
                                  state="readonly", width=10)
        model_combo.pack(side=tk.LEFT)
        ttk.Label(model_frame, text="(tiny=fastest, large=most accurate)").pack(side=tk.LEFT, padx=(10, 0))

        # Boolean settings
        ttk.Label(main_frame, text="Options:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(15, 5))

        self.always_on_top_var = tk.BooleanVar(value=self.settings.get('always_on_top', False))
        ttk.Checkbutton(main_frame, text="Keep window always on top",
                       variable=self.always_on_top_var).pack(anchor=tk.W, pady=2)

        self.minimize_to_tray_var = tk.BooleanVar(value=self.settings.get('minimize_to_tray', True))
        ttk.Checkbutton(main_frame, text="Minimize to system tray when closed",
                       variable=self.minimize_to_tray_var).pack(anchor=tk.W, pady=2)

        self.auto_copy_var = tk.BooleanVar(value=self.settings.get('auto_copy_clipboard', True))
        ttk.Checkbutton(main_frame, text="Automatically copy processed text to clipboard",
                       variable=self.auto_copy_var).pack(anchor=tk.W, pady=2)

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(20, 0))

        ttk.Button(button_frame, text="Cancel", command=self.cancel).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(button_frame, text="Apply", command=self.apply).pack(side=tk.RIGHT)

    def apply(self):
        """Apply the settings"""
        # Update settings
        self.settings.set('hotkey', self.hotkey_var.get())
        self.settings.set('whisper_model', self.model_var.get())
        self.settings.set('always_on_top', self.always_on_top_var.get())
        self.settings.set('minimize_to_tray', self.minimize_to_tray_var.get())
        self.settings.set('auto_copy_clipboard', self.auto_copy_var.get())

        # Call the apply callback
        if self.apply_callback:
            self.apply_callback()

        self.dialog.destroy()

    def cancel(self):
        """Cancel the dialog"""
        self.dialog.destroy()

class VoiceRecorderGUI:
    """GUI version of the voice recorder with hotkey support"""

    def __init__(self):
        self.settings = SettingsManager()
        self.recorder = VoiceRecorder(settings_manager=self.settings)
        self.is_recording = False
        self.hotkey_listener = None
        self.tray_icon = None
        self.is_closing = False

        # Create main window
        self.root = tk.Tk()
        self.root.title("Voice to Text Converter")
        geometry = self.settings.get('window_geometry', '600x500')
        self.root.geometry(geometry)
        self.root.resizable(True, True)

        # Set always on top if enabled
        if self.settings.get('always_on_top', False):
            self.root.wm_attributes('-topmost', True)

        # Set up tray first (before UI setup)
        if self.settings.get('minimize_to_tray', True):
            self.setup_tray()

        self.setup_ui()
        self.setup_hotkey()

    def setup_ui(self):
        """Set up the GUI elements"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)

        # Title
        title_label = ttk.Label(main_frame, text="Voice to Text Converter",
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # Record button
        self.record_button = ttk.Button(main_frame, text="🎤 Record",
                                       command=self.toggle_recording,
                                       style="Record.TButton")
        self.record_button.grid(row=1, column=0, columnspan=2, pady=10, sticky="ew")

        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready to record (Press F1 or click Record)",
                                     font=('Arial', 10))
        self.status_label.grid(row=2, column=0, columnspan=2, pady=(0, 10))

        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Transcription Results", padding="10")
        results_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)

        # Raw transcription
        ttk.Label(results_frame, text="Raw Transcription:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky=tk.W)
        self.raw_text = scrolledtext.ScrolledText(results_frame, height=6, width=70)
        self.raw_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(5, 10))

        # Processed transcription
        ttk.Label(results_frame, text="Processed Prompt (Copied to Clipboard):", font=('Arial', 10, 'bold')).grid(row=2, column=0, sticky=tk.W)
        self.processed_text = scrolledtext.ScrolledText(results_frame, height=6, width=70)
        self.processed_text.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(5, 0))

        # Control buttons frame
        controls_frame = ttk.Frame(main_frame)
        controls_frame.grid(row=4, column=0, columnspan=2, pady=10, sticky="ew")
        controls_frame.columnconfigure(0, weight=1)
        controls_frame.columnconfigure(1, weight=1)
        controls_frame.columnconfigure(2, weight=1)

        # Always on top toggle
        self.always_on_top_var = tk.BooleanVar(value=self.settings.get('always_on_top', False))
        always_on_top_cb = ttk.Checkbutton(controls_frame, text="Always on Top",
                                          variable=self.always_on_top_var,
                                          command=self.toggle_always_on_top)
        always_on_top_cb.grid(row=0, column=0, sticky="w")

        # Settings button
        settings_btn = ttk.Button(controls_frame, text="⚙️ Settings",
                                 command=self.open_settings)
        settings_btn.grid(row=0, column=1, padx=5)

        # Minimize to tray button (if tray enabled and available)
        if self.settings.get('minimize_to_tray', True) and self.tray_icon:
            tray_btn = ttk.Button(controls_frame, text="📌 Minimize to Tray",
                                 command=self.minimize_to_tray)
            tray_btn.grid(row=0, column=2, sticky="e")

        # Hotkey info
        hotkey = self.settings.get('hotkey', 'f1').upper()
        info_label = ttk.Label(main_frame, text=f"💡 Tip: Press {hotkey} anywhere to start/stop recording",
                              font=('Arial', 9), foreground="gray")
        info_label.grid(row=5, column=0, columnspan=2, pady=10)

        # Configure button style
        style = ttk.Style()
        style.configure("Record.TButton", font=('Arial', 12, 'bold'))

    def setup_hotkey(self):
        """Set up global hotkey listener"""
        def on_hotkey():
            # Schedule the toggle in the main thread
            self.root.after(0, self.toggle_recording)

        # Get hotkey from settings
        hotkey = self.settings.get('hotkey', 'f1')

        # Start hotkey listener in background thread
        self.hotkey_listener = keyboard.GlobalHotKeys({
            f'<{hotkey}>': on_hotkey
        })
        self.hotkey_listener.start()

    def setup_tray(self):
        """Set up system tray icon"""
        try:
            # Create a simple icon (avoid emoji text which can cause issues)
            image = Image.new('RGB', (64, 64), color='blue')
            draw = ImageDraw.Draw(image)
            draw.ellipse([16, 16, 48, 48], fill='white')
            draw.ellipse([24, 24, 40, 40], fill='blue')  # Simple microphone representation

            # Create tray menu
            menu = pystray.Menu(
                pystray.MenuItem('Show', self.show_window),
                pystray.MenuItem('Record', self.toggle_recording),
                pystray.MenuItem('Settings', self.open_settings),
                pystray.MenuItem('Quit', self.quit_app)
            )

            self.tray_icon = pystray.Icon('VoiceToText', image, 'Voice to Text', menu)
        except Exception as e:
            print(f"Warning: Could not set up system tray: {e}")
            print("System tray features will be disabled.")
            self.tray_icon = None
            # Disable tray setting if it fails
            self.settings.set('minimize_to_tray', False)

    def toggle_always_on_top(self):
        """Toggle always on top setting"""
        always_on_top = self.always_on_top_var.get()
        self.root.wm_attributes('-topmost', always_on_top)
        self.settings.set('always_on_top', always_on_top)

    def minimize_to_tray(self):
        """Minimize window to system tray"""
        if self.tray_icon:
            self.root.withdraw()
            # Start tray icon in background thread
            threading.Thread(target=self.tray_icon.run, daemon=True).start()
        else:
            # If tray is not available, just minimize normally
            self.root.iconify()
            messagebox.showinfo("Minimized", "Window minimized to taskbar (system tray not available)")

    def show_window(self, icon=None, item=None):
        """Show window from tray"""
        self.root.deiconify()
        self.root.lift()
        if self.tray_icon:
            self.tray_icon.stop()

    def open_settings(self, icon=None, item=None):
        """Open settings dialog"""
        SettingsDialog(self.root, self.settings, self.apply_settings)

    def apply_settings(self):
        """Apply new settings to the application"""
        # Update hotkey
        if self.hotkey_listener:
            self.hotkey_listener.stop()
        self.setup_hotkey()

        # Update always on top
        always_on_top = self.settings.get('always_on_top', False)
        self.always_on_top_var.set(always_on_top)
        self.root.wm_attributes('-topmost', always_on_top)

        # Update hotkey info label
        hotkey = self.settings.get('hotkey', 'f1').upper()
        # Find and update the info label (this is a bit hacky but works)
        for widget in self.root.winfo_children():
            for child in widget.winfo_children():
                if isinstance(child, ttk.Label) and '💡 Tip:' in child.cget('text'):
                    child.config(text=f"💡 Tip: Press {hotkey} anywhere to start/stop recording")

    def quit_app(self, icon=None, item=None):
        """Quit the application completely"""
        self.is_closing = True
        if self.tray_icon:
            self.tray_icon.stop()
        if self.hotkey_listener:
            self.hotkey_listener.stop()
        self.root.quit()
        self.root.destroy()

    def toggle_recording(self):
        """Toggle recording state"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        """Start recording in background thread"""
        if self.is_recording:
            return

        self.is_recording = True
        self.record_button.config(text="🛑 Stop Recording", style="Stop.TButton")
        self.status_label.config(text="🔴 Recording... Click Stop or press F1 to finish")

        # Configure stop button style
        style = ttk.Style()
        style.configure("Stop.TButton", font=('Arial', 12, 'bold'), foreground="red")

        # Clear previous results
        self.raw_text.delete(1.0, tk.END)
        self.processed_text.delete(1.0, tk.END)

        # Start recording in background thread
        threading.Thread(target=self._record_audio, daemon=True).start()

    def _record_audio(self):
        """Background recording method"""
        try:
            # Start recording
            self.recorder.recording = True
            self.recorder.audio_data = []

            def callback(indata, frames, time, status):
                if self.recorder.recording:
                    self.recorder.audio_data.append(indata.copy())

            # Update status in main thread
            self.root.after(0, lambda: self.status_label.config(text="🔴 Recording... Speak now!"))

            with sd.InputStream(callback=callback,
                              samplerate=self.recorder.sample_rate,
                              channels=self.recorder.channels,
                              dtype=np.float32):
                # Wait until recording is stopped
                while self.is_recording:
                    threading.Event().wait(0.1)

        except Exception as e:
            self.root.after(0, lambda: self._handle_recording_error(str(e)))

    def stop_recording(self):
        """Stop recording and process audio"""
        if not self.is_recording:
            return

        self.is_recording = False
        self.recorder.recording = False

        self.record_button.config(text="🎤 Record", style="Record.TButton")
        self.status_label.config(text="⏳ Processing transcription...")

        # Process audio in background thread
        threading.Thread(target=self._process_audio, daemon=True).start()

    def _process_audio(self):
        """Process recorded audio and update GUI"""
        try:
            if not self.recorder.audio_data:
                self.root.after(0, lambda: self._handle_recording_error("No audio recorded"))
                return

            # Combine audio data
            audio = np.concatenate(self.recorder.audio_data, axis=0)

            # Update status
            self.root.after(0, lambda: self.status_label.config(text="🤖 Loading Whisper model..."))

            # Load model and transcribe
            model_name = self.settings.get('whisper_model', 'base') if hasattr(self, 'settings') else 'base'
            model = whisper.load_model(model_name)
            temp_file = self.recorder._save_to_temp_file(audio)

            self.root.after(0, lambda: self.status_label.config(text="🤖 Transcribing audio..."))

            try:
                result = model.transcribe(temp_file)
                raw_text = result["text"]
                processed_text = self.recorder.processor.process(raw_text)

                # Update GUI in main thread
                self.root.after(0, lambda: self._update_results(raw_text, processed_text))

                # Save to file
                self._save_transcription(raw_text, processed_text)

            finally:
                os.unlink(temp_file)

        except Exception as e:
            self.root.after(0, lambda: self._handle_recording_error(str(e)))

    def _update_results(self, raw_text, processed_text):
        """Update GUI with transcription results"""
        # Update text widgets
        self.raw_text.delete(1.0, tk.END)
        self.raw_text.insert(1.0, raw_text)

        self.processed_text.delete(1.0, tk.END)
        self.processed_text.insert(1.0, processed_text)

        # Copy to clipboard if enabled
        if self.settings.get('auto_copy_clipboard', True):
            try:
                pyperclip.copy(processed_text)
                self.status_label.config(text="✅ Transcription complete! Processed prompt copied to clipboard.")
            except Exception as e:
                self.status_label.config(text="✅ Transcription complete! (Clipboard copy failed)")
        else:
            self.status_label.config(text="✅ Transcription complete!")

    def _save_transcription(self, raw_text, processed_text):
        """Save transcription to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.recorder.transcripts_dir / f"transcription_{timestamp}.txt"

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Transcription - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*60 + "\n")
                f.write("RAW TRANSCRIPTION:\n")
                f.write("="*60 + "\n")
                f.write(raw_text + "\n\n")
                f.write("="*60 + "\n")
                f.write("PROCESSED PROMPT:\n")
                f.write("="*60 + "\n")
                f.write(processed_text)

        except Exception as e:
            print(f"Error saving file: {e}")

    def _handle_recording_error(self, error_msg):
        """Handle recording errors"""
        self.is_recording = False
        self.recorder.recording = False
        self.record_button.config(text="🎤 Record", style="Record.TButton")
        self.status_label.config(text=f"❌ Error: {error_msg}")
        messagebox.showerror("Recording Error", error_msg)

    def run(self):
        """Start the GUI application"""
        try:
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.root.bind('<Configure>', lambda e: self.save_window_geometry() if e.widget == self.root else None)
            self.root.mainloop()
        finally:
            self.save_window_geometry()
            if self.hotkey_listener:
                self.hotkey_listener.stop()
            if self.tray_icon:
                self.tray_icon.stop()

    def on_closing(self):
        """Handle window closing"""
        if self.settings.get('minimize_to_tray', True) and not self.is_closing and self.tray_icon:
            # Minimize to tray instead of closing (only if tray is available)
            self.minimize_to_tray()
        else:
            # Actually close
            self.quit_app()

    def save_window_geometry(self):
        """Save current window geometry"""
        try:
            geometry = self.root.geometry()
            self.settings.set('window_geometry', geometry)
        except Exception:
            pass

def main_terminal():
    """Terminal version of the voice recorder"""
    recorder = VoiceRecorder()

    print("Voice to Text Converter (Terminal Mode)")
    print("=======================================")

    while True:
        print("\n1. Record (Enter to stop)")
        print("2. Quit")

        choice = input("\nSelect option (1-2): ").strip()

        if choice == "1":
            recorder.record_audio()
        elif choice == "2":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please select 1 or 2.")

def main():
    """Main entry point - check for UI argument"""
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'ui':
        # Launch GUI version
        print("Starting Voice to Text Converter (GUI Mode)...")
        app = VoiceRecorderGUI()
        app.run()
    else:
        # Launch terminal version
        main_terminal()

if __name__ == "__main__":
    main()