"""
Export utilities for transcription results

Supports multiple export formats: TXT, SRT, JSON, TSV, VTT
"""

import json
from datetime import timedelta
from pathlib import Path
from typing import Dict, List


class TranscriptionExporter:
    """Export transcription results in various formats"""

    @staticmethod
    def export_txt(result: Dict, file_path: Path) -> None:
        """
        Export transcription as plain text file.

        Args:
            result: Transcription result dictionary
            file_path: Output file path
        """
        text = result.get("full_text", "") or result.get("text", "")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)

    @staticmethod
    def export_srt(result: Dict, file_path: Path) -> None:
        """
        Export transcription as SRT subtitle file.

        Args:
            result: Transcription result dictionary
            file_path: Output file path
        """
        segments = result.get("segments", [])

        with open(file_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(segments, 1):
                start = TranscriptionExporter._format_srt_time(segment.get("start", 0))
                end = TranscriptionExporter._format_srt_time(segment.get("end", 0))
                text = segment.get("text", "").strip()

                if text:
                    f.write(f"{i}\n")
                    f.write(f"{start} --> {end}\n")
                    f.write(f"{text}\n\n")

    @staticmethod
    def export_vtt(result: Dict, file_path: Path) -> None:
        """
        Export transcription as WebVTT subtitle file.

        Args:
            result: Transcription result dictionary
            file_path: Output file path
        """
        segments = result.get("segments", [])

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")

            for segment in segments:
                start = TranscriptionExporter._format_vtt_time(segment.get("start", 0))
                end = TranscriptionExporter._format_vtt_time(segment.get("end", 0))
                text = segment.get("text", "").strip()

                if text:
                    f.write(f"{start} --> {end}\n")
                    f.write(f"{text}\n\n")

    @staticmethod
    def export_json(result: Dict, file_path: Path) -> None:
        """
        Export transcription as JSON file.

        Args:
            result: Transcription result dictionary
            file_path: Output file path
        """
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    @staticmethod
    def export_tsv(result: Dict, file_path: Path) -> None:
        """
        Export transcription as TSV (tab-separated values) file.

        Args:
            result: Transcription result dictionary
            file_path: Output file path
        """
        segments = result.get("segments", [])

        with open(file_path, "w", encoding="utf-8") as f:
            # Write header
            f.write("Index\tStart\tEnd\tDuration\tText\n")

            for i, segment in enumerate(segments, 1):
                start = segment.get("start", 0)
                end = segment.get("end", 0)
                duration = end - start
                text = segment.get("text", "").strip()

                if text:
                    f.write(
                        f"{i}\t{start:.2f}\t{end:.2f}\t{duration:.2f}\t{text}\n"
                    )

    @staticmethod
    def export(
        result: Dict, file_path: Path, format_type: str = "txt"
    ) -> None:
        """
        Export transcription in specified format.

        Args:
            result: Transcription result dictionary
            file_path: Output file path
            format_type: Export format ('txt', 'srt', 'vtt', 'json', 'tsv')

        Raises:
            ValueError: If format is not supported
        """
        format_type = format_type.lower()

        exporters = {
            "txt": TranscriptionExporter.export_txt,
            "srt": TranscriptionExporter.export_srt,
            "vtt": TranscriptionExporter.export_vtt,
            "json": TranscriptionExporter.export_json,
            "tsv": TranscriptionExporter.export_tsv,
        }

        if format_type not in exporters:
            raise ValueError(
                f"Unsupported format: {format_type}. "
                f"Supported formats: {list(exporters.keys())}"
            )

        exporters[format_type](result, file_path)

    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        """Format time for SRT format (HH:MM:SS,mmm)"""
        td = timedelta(seconds=seconds)
        hours, remainder = divmod(int(td.total_seconds()), 3600)
        minutes, secs = divmod(remainder, 60)
        milliseconds = int((seconds % 1) * 1000)

        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

    @staticmethod
    def _format_vtt_time(seconds: float) -> str:
        """Format time for VTT format (HH:MM:SS.mmm)"""
        td = timedelta(seconds=seconds)
        hours, remainder = divmod(int(td.total_seconds()), 3600)
        minutes, secs = divmod(remainder, 60)
        milliseconds = int((seconds % 1) * 1000)

        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"
