from .summarizer import SummarizerQWEN, clip_text

def run_summarization_pipeline(topics):
    summarizer = SummarizerQWEN()
    chapters = []

    for i, t in enumerate(topics):
        full_text = t["text"]
        word_count = len(full_text.split())

        s = summarizer.summarize_topic(full_text)


        if not s["summary"].strip():
            s = summarizer.summarize_long_topic(full_text)


        if not s["summary"].strip():
            clipped = clip_text(full_text, max_words=350)
            s = summarizer.summarize_topic(clipped)


        chapters.append({
            "topic_id": i,
            "start": t["start"],
            "end": t["end"],
            "text": full_text,
            **s,
        })

    return chapters
