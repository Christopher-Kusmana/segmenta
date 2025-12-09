import re

def clean_text(text: str) -> str:
    """Basic cleanup: remove extra spaces, normalize punctuation, etc."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()
