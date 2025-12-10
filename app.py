import gradio as gr
import os
from pathlib import Path
from src.unified.unified_pipeline import run_unified_pipeline
import traceback

def process_video(video_file, youtube_url):
    """
    Full backend logic:
    1. Run STT → transcript segments
    2. Run NLP → chapters, topics
    3. Format for UI
    """

    try:
        if video_file is None:
            return "⚠️ Please upload a video file.", "", [], ""

        # -----------------------
        # Run full unified pipeline
        # -----------------------
        results = run_unified_pipeline(
            video_file=video_file,
            save_transcript=False
        )

        # Extract results
        transcript_text = results["transcript_text"]
        chapters = results["chapters"]

        # -----------------------
        # Convert chapters → table rows
        # -----------------------
        chapter_rows = []
        for chap in chapters:
            timestamp = f"{chap['start']:.2f} → {chap['end']:.2f}"
            title = chap.get("title", "(no title)")
            summary = chap.get("summary", "")
            chapter_rows.append([timestamp, title, summary])

        status = "✅ Video processed successfully!"

        return status, transcript_text, chapter_rows, ""

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

with gr.Blocks(title="AI Video Lecture Analyzer") as app:
    
    gr.Markdown("""
    # 🎓 AI Video Lecture Analyzer
    **Automatic Topic Segmentation • Smart Summaries • Semantic Search**
    
    Upload a video or paste a YouTube URL to get started.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Input")
            
            video_input = gr.Video(
                label="Upload Video File",
                sources=["upload"]
            )
            
            youtube_input = gr.Textbox(
                label="Or paste YouTube URL",
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
                        lines=15,
                        interactive=False
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
    - **Phase 1**: Whisper speech-to-text integration
    - **Phase 2**: Sentence embeddings + topic segmentation
    - **Phase 3**: LLM-powered chapter titles & summaries
    - **Phase 4**: Vector database + semantic search
    - **Phase 5**: Polish & advanced features
    """)
    
    # Event handlers
    process_btn.click(
        fn=process_video,
        inputs=[video_input, youtube_input],
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

