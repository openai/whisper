import os
import torch
from pyannote.audio import Pipeline
import threading

# Global Diarization Pipeline (Lazy Loaded)
_pipeline_instance = None
_pipeline_lock = threading.Lock()

def get_diarization_pipeline(use_auth_token=None):
    global _pipeline_instance
    with _pipeline_lock:
        if _pipeline_instance is None:
            print("Loading Diarization pipeline...")
            # Use the standard pretrained model
            # Note: This requires an access token for 'pyannote/speaker-diarization-3.1'
            # If not provided, it will look for HF_TOKEN env var
            try:
                _pipeline_instance = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=use_auth_token
                )
                
                if _pipeline_instance:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    _pipeline_instance.to(device)
                    print(f"Diarization pipeline loaded on {device}")
                else:
                    print("Failed to load diarization pipeline. Check HF_TOKEN.")
            except Exception as e:
                print(f"Error loading diarization pipeline: {e}")
                raise e
                
    return _pipeline_instance

def run_diarization(file_path, num_speakers=None):
    """
    Run diarization on an audio file.
    Returns a list of segments: [{'start': 0.0, 'end': 1.5, 'speaker': 'SPEAKER_00'}, ...]
    """
    pipeline = get_diarization_pipeline()
    if not pipeline:
        raise RuntimeError("Diarization pipeline not initialized")
        
    # Run inference
    diarization = pipeline(file_path, num_speakers=num_speakers)
    
    # Format results
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })
        
    return segments

def merge_transcription_with_diarization(transcription_result, diarization_segments):
    """
    Assign speakers to transcription segments based on time overlap.
    """
    transcript_segments = transcription_result.get("segments", [])
    
    for t_seg in transcript_segments:
        t_start = t_seg["start"]
        t_end = t_seg["end"]
        
        # Find all diarization segments that overlap with this transcription segment
        overlaps = []
        for d_seg in diarization_segments:
            # Calculate overlap duration
            start = max(t_start, d_seg["start"])
            end = min(t_end, d_seg["end"])
            duration = max(0, end - start)
            
            if duration > 0:
                overlaps.append((d_seg["speaker"], duration))
        
        # Assign the speaker with the most overlap
        if overlaps:
            # Sort by duration descending
            overlaps.sort(key=lambda x: x[1], reverse=True)
            best_speaker = overlaps[0][0]
            t_seg["speaker"] = best_speaker
        else:
            t_seg["speaker"] = "Unknown"
            
    return transcription_result
