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


# ======================================================
# FULL NLP PIPELINE TEST (Notebook)
# ======================================================

from src.nlp.pipeline_partition import run_nlp_partition_pipeline
from src.nlp.pipeline_summarization import run_summarization_pipeline

print("▶ Running NLP Partitioning...\n")
partition_results = run_nlp_partition_pipeline(
    whisper_segments=segments,
    max_duration=60,
    max_gap=1.5
)

chunks = partition_results["chunks"]
topics = partition_results["topics"]
boundaries = partition_results["boundaries"]

print("🧩 Chunks:", len(chunks))
print("📌 Detected boundaries:", boundaries)
print("📘 Topics:", len(topics))

print("\n▶ Running Summarization...\n")
chapters = run_summarization_pipeline(topics)

print(f"Generated {len(chapters)} chapter summaries.\n")

# Pretty print summaries
for chap in chapters:
    print("=" * 100)
    print(f"📌 Topic {chap['topic_id'] + 1}")
    print(f"⏱  {chap['start']} → {chap['end']}\n")
    print(f"🎬 Title:\n{chap['title']}\n")
    print(f"📝 Summary:\n{chap['summary']}\n")
    print("🔹 Bullet Points:")
    for b in chap["bullets"]:
        print(f" - {b}")
    print("\n🔑 Keywords:", ", ".join(chap["keywords"]))
    print("\n")
