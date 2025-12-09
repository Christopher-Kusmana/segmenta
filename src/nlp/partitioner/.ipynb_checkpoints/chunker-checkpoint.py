from typing import List, Dict

def merge_whisper_segments(
    segments: List[Dict],
    max_gap: float = 1.5,
    max_duration: float = 90.0
):
    """
    Merge whisper segments into larger blocks for stable embeddings.

    Rules:
    - Merge consecutive segments if time gap <= max_gap
    - Stop merging when block exceeds max_duration
    """

    merged = []
    current = []

    for seg in segments:
        if not current:
            current.append(seg)
            continue

        last = current[-1]
        gap = seg["start"] - last["end"]
        duration = seg["end"] - current[0]["start"]

        if gap <= max_gap and duration <= max_duration:
            current.append(seg)
        else:
            merged.append({
                "start": current[0]["start"],
                "end": current[-1]["end"],
                "text": " ".join([s["text"] for s in current])
            })
            current = [seg]

    if current:
        merged.append({
            "start": current[0]["start"],
            "end": current[-1]["end"],
            "text": " ".join([s["text"] for s in current])
        })

    return merged
