from huggingface_hub import InferenceClient
from .prompts import FULL_SUMMARY_PROMPT

def clip_text(text: str, max_words: int = 450) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])



def chunk_text(text: str, chunk_size: int = 350):
    """
    Split long transcripts into manageable chunks for summarization.
    """
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))

    return chunks
    

class SummarizerQWEN:
    """
    Handles ALL summarization:
    - title
    - summary
    - bullets
    - keywords
    using a single LLM call (Qwen2.5-1.5B).
    """

    def __init__(self, model="Qwen/Qwen2.5-1.5B-Instruct"):
        self.client = InferenceClient(model)

    def summarize_topic(self, text: str):
        prompt = FULL_SUMMARY_PROMPT.format(topic_text=text)

        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.3,
        )

        raw = response.choices[0].message["content"].strip()

        # Parse using simple tags
        return self._parse_output(raw)

    def _parse_output(self, text):
        """
        Expect the model to return structured text in this format:

        Title: ...
        Summary: ...
        Bullets:
        - ...
        - ...
        Keywords: kw1, kw2, kw3
        """

        title = self._extract(text, "Title")
        summary = self._extract(text, "Summary")
        bullets = self._extract_list(text, "Bullets")
        keywords = self._extract_csv(text, "Keywords")

        return {
            "title": title,
            "summary": summary,
            "bullets": bullets,
            "keywords": keywords,
        }

    def _extract(self, text, label):
        import re
        pattern = rf"{label}:\s*(.*)"
        m = re.search(pattern, text)
        return m.group(1).strip() if m else ""

    def _extract_list(self, text, label):
        import re
        section = re.search(rf"{label}:(.*?)(?:\n\n|$)", text, re.S)
        if not section:
            return []
        lines = section.group(1).strip().split("\n")
        return [l.strip("-• ").strip() for l in lines if l.strip()]

    def _extract_csv(self, text, label):
        import re
        m = re.search(rf"{label}:\s*(.*)", text)
        if not m:
            return []
        return [k.strip() for k in m.group(1).split(",")]

    def summarize_long_topic(self, text: str):
        """
        For long segments:
        1. Chunk text
        2. Summarize each chunk individually
        3. Summarize the summaries into a final combined summary
        """

        chunks = chunk_text(text, chunk_size=350)
    
        partial_summaries = []
        for c in chunks:
            s = self.summarize_topic(c)
            partial_summaries.append(s["summary"])
    
        merged = "\n".join(partial_summaries)
        final = self.summarize_topic(merged)
        return final