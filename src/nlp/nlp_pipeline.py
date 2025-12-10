from .partitioner.pipeline import run_nlp_partition_pipeline
from .summarizer.pipeline import run_summarization_pipeline

def run_full_nlp_pipeline(whisper_segments):
    """
    Full NLP pipeline:
      1. Segment merged transcript into topics
      2. Summarize each topic
    """

    # Topic segmentation
    part_results = run_nlp_partition_pipeline(
        whisper_segments=whisper_segments,
        max_duration=60,
        max_gap=2.0,
    )
    
    topics = part_results["topics"]

    # Summarization
    chapters = run_summarization_pipeline(topics)

    return {
        "chunks": part_results["chunks"],
        "topics": topics,
        "chapters": chapters,
        "boundaries": part_results["boundaries"],
        "raw_boundaries": part_results["raw_boundaries"],
    }
