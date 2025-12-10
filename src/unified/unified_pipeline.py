"""
Unified Pipeline: STT + NLP
---------------------------
This file orchestrates the entire flow:

1. STT (Whisper) → transcript segments
2. NLP (Partition + Summarization) → chapters, topics
3. return all structured outputs in one dictionary
"""

from pathlib import Path
import os

from src.stt.stt_pipeline import run_stt_pipeline
from src.nlp.nlp_pipeline import run_full_nlp_pipeline


def run_unified_pipeline(video_file=None, save_transcript=False):
    """
    Full end-to-end pipeline:
        VIDEO → STT → Transcript → NLP → Topics → Summaries

    Parameters
    ----------
    video_file : path-like or file object
        Uploaded video file (Gradio)
    youtube_url : str
        Optional. If provided, download audio and run STT.
    save_transcript : bool
        If True, saves the transcript JSON for debugging.

    Returns
    -------
    dict
        {
            "segments": [...],      # raw whisper segments
            "transcript_text": "...",
            "chunks": [...],
            "topics": [...],
            "chapters": [...],
            "boundaries": [...],
            "raw_boundaries": [...]
        }
    """

    # STT Pipeline
    
    stt_output = run_stt_pipeline(
        video_file=video_file,
        save_output=save_transcript
    )

    segments = stt_output["segments"]
    transcript_text = stt_output["transcript_text"]

    # NLP Pipeline

    nlp_output = run_full_nlp_pipeline(segments)

    # Merge result
    
    result = {
        "segments": segments,
        "transcript_text": transcript_text,
        "chunks": nlp_output["chunks"],
        "topics": nlp_output["topics"],
        "chapters": nlp_output["chapters"],
        "boundaries": nlp_output["boundaries"],
        "raw_boundaries": nlp_output["raw_boundaries"],
    }

    return result
