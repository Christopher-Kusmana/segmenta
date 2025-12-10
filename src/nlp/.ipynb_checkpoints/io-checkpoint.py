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