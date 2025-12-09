from typing import List, Dict
from .chunker import merge_whisper_segments
from .embedder import Embedder
from .text_utils import clean_text
from .segmenter import topic_segmentation


import os 
import sys


PROJECT_ROOT = os.path.abspath("..")
sys.path.append(PROJECT_ROOT)

PROJECT_ROOT

import json
from pathlib import Path

def load_transcript_json(path: str):
    """
    Load whisper transcript JSON and return list of segments.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Transcript JSON not found: {path}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data 



def run_nlp_partition_pipeline(
    whisper_segments,
    max_duration=60,
    max_gap=1.5,
):
    chunks = merge_whisper_segments(
        whisper_segments,
        max_duration=max_duration,
        max_gap=max_gap
    )

    embedder = Embedder()
    texts = [clean_text(c["text"]) for c in chunks]
    embeddings = embedder.encode(texts)

    results = topic_segmentation(chunks, embeddings)

    return {
        "chunks": chunks,
        **results
    }
