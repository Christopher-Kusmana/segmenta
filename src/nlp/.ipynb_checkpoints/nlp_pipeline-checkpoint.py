from .partitioner.pipeline import run_nlp_partition_pipeline
from .summarizer.pipeline import run_summarization_pipeline

def run_full_nlp_pipeline(transcript_json):
    """
    1. Partition transcript into topic segments
    2. Summarize each topic into title, summary, bullets, keywords
    """

    # Segmentation
    part_results = run_nlp_partition_pipeline(transcript_json)
    topics = part_results["topics"]

    # Summarization (HF inference)
    chapters = run_summarization_pipeline(topics)

    return {
        "chunks": part_results["chunks"],
        "topics": topics,
        "chapters": chapters,
        "boundaries": part_results["boundaries"],
        "raw_boundaries": part_results["raw_boundaries"]
    }
