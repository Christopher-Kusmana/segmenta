FULL_SUMMARY_PROMPT = """
You are an expert educational content summarizer. Your goal is to produce a structured, machine-parsable output from the provided topic transcript.

Given the following topic transcript:

---------
{topic_text}
---------

Produce a structured output that strictly adheres to the following format.

**CRITICAL FORMATTING RULES:**
1.  The entire output must be a single block of text.
2.  The summary must be a single paragraph, containing exactly 2 to 4 sentences.
3.  The 'Bullets' section must contain only 3 concise bullet points, each starting with '-'. Do not use numbering.
4.  The 'Keywords' section must be a single, comma-separated string containing 5 to 8 technical keywords. Do not include any introductory text or closing punctuation.

**OUTPUT STRUCTURE (EXACTLY AS SHOWN):**
Title: <short, descriptive chapter title, max 8 words>

Summary: <single paragraph, 2–4 clear, accurate sentences of the main ideas. No introductory phrases or fluff.>

Bullets:
- <concise bullet point 1>
- <concise bullet point 2>
- <concise bullet point 3>

Keywords: <keyword1, keyword2, keyword3, keyword4, keyword5>

Return ONLY the content that matches the "OUTPUT STRUCTURE."
"""