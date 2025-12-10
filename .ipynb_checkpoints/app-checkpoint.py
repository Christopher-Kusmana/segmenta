import gradio as gr
import os
from pathlib import Path
from src.unified.unified_pipeline import run_unified_pipeline
import traceback


def format_timestamp(sec):
    minutes = int(sec // 60)
    seconds = int(sec % 60)
    return f"{minutes:02d}:{seconds:02d}"

def seconds_to_mmss(seconds: float) -> str:
    seconds = int(seconds)
    m = seconds // 60
    s = seconds % 60
    return f"{m:02d}:{s:02d}"


def process_video(video_file, youtube_url, progress=gr.Progress()):
    try:
        if video_file is None:
            return "⚠️ Please upload a video file.", "", [], ""

        # 0–10%: STT Loading
        progress(0.05, desc="Initializing Whisper engine...")
        progress(0.10, desc="Transcribing video (this one takes the longest 😴)")

        # Run full pipeline (STT + NLP)
        results = run_unified_pipeline(
            video_file=video_file,
            save_transcript=False
        )

        # 10–60%: Topic segmentation
        progress(0.60, desc="Detecting topic boundaries...")

        segments = results["segments"]
        transcript_text = results["transcript_text"]

        topics = results["topics"]

        # 60–95%: LLM summarization
        progress(0.95, desc="Generating chapter summaries...")

        chapters = results["chapters"]

        # Build output rows
        chapter_rows = []
        for chap in chapters:
            start = chap["start"]
            end = chap["end"]
            ts = f"{seconds_to_mmss(chap['start'])} → {seconds_to_mmss(chap['end'])}"

            chapter_rows.append([
                ts,
                chap.get("title", "(no title)"),
                chap.get("summary", "")
            ])

        progress(1.0, desc="Done! ✓")

        return "✅ Video processed successfully!", transcript_text, chapter_rows, ""

    except Exception as e:
        traceback.print_exc()
        return f"❌ Error processing video: {str(e)}", "", [], ""



def search_transcript(query, transcript):
    """
    Semantic search function - will use vector embeddings in future phases
    Currently does simple text matching for testing
    """
    if not query or not transcript:
        return "No search query or transcript available"
    
    # Simple mock search (will be replaced with semantic search)
    lines = transcript.split('\n')
    matches = [line for line in lines if query.lower() in line.lower()]
    
    if matches:
        result = f"🔍 Found {len(matches)} matches for '{query}':\n\n"
        result += "\n".join(matches)
    else:
        result = f"No matches found for '{query}'"
    
    return result


# ============================================================================
# Gradio Interface
# ============================================================================

custom_css = """
#transcript_box textarea {
    overflow-y: auto !important;
    height: 400px !important;
}
"""

with gr.Blocks(title="Segmenta | AI Lecture Video Analyzer") as app:
    
    gr.Markdown("""
    # 🎓 Segmenta: AI Lecture Video Analyzer
    **Automatic Topic Segmentation • Smart Summaries • Semantic Search**
    
    Upload a video or paste a YouTube URL (not implemented yet) to get started.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Input")
            
            video_input = gr.Video(
                label="Upload Video File",
                sources=["upload"]
            )
            
            youtube_input = gr.Textbox(
                label="Or paste YouTube URL (to be implemented...)",
                placeholder="https://www.youtube.com/watch?v=..."
            )
            
            process_btn = gr.Button("🚀 Process Video", variant="primary", size="lg")
            
            status_output = gr.Textbox(
                label="Status",
                interactive=False,
                lines=2
            )
        
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Results")
            
            with gr.Tabs():
                with gr.Tab("📑 Chapters"):
                    chapters_output = gr.Dataframe(
                        headers=["Timestamp", "Title", "Summary"],
                        label="Video Chapters",
                        interactive=False,
                        wrap=True
                    )
                
                with gr.Tab("📝 Transcript"):
                    transcript_output = gr.Textbox(
                        label="Full Transcript",
                        lines=20,
                        interactive=True,
                        elem_id="transcript_box"
                    )
                
                with gr.Tab("🔍 Search"):
                    with gr.Row():
                        search_input = gr.Textbox(
                            label="Search Query",
                            placeholder="Enter keywords to search (e.g., 'gradient descent', 'activation function')"
                        )
                        search_btn = gr.Button("Search", variant="secondary")
                    
                    search_output = gr.Textbox(
                        label="Search Results",
                        lines=10,
                        interactive=False
                    )
    
    gr.Markdown("""
    ---
    ### 🛠️ Implementation Roadmap
    - **Phase 0** (Current): Basic UI framework ✅
    - **Phase 1**: Whisper speech-to-text integration ✅
    - **Phase 2**: Sentence embeddings + topic segmentation ✅
    - **Phase 3**: LLM-powered chapter titles & summaries ✅
    - **Phase 4**: Vector database + semantic search ✅
    - **Phase 5**: Post Processing + Advanced Features
    """)
    
    # Event handlers
    process_btn.click(
        fn=process_video,
        inputs=[video_input],
        outputs=[status_output, transcript_output, chapters_output, search_output]
    )

    
    search_btn.click(
        fn=search_transcript,
        inputs=[search_input, transcript_output],
        outputs=search_output
    )

# ============================================================================
# Launch
# ============================================================================

if __name__ == "__main__":
    app.launch(theme=gr.themes.Soft())

