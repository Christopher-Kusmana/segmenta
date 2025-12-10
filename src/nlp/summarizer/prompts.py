FULL_SUMMARY_PROMPT = """
You are an expert educational content summarizer specializing in technical lectures. Your goal is to produce a structured, machine-parsable output that clearly separates the conceptual stages of the lecture.

Given the following topic transcript:

---------
{topic_text}
---------

INSTRUCTION BLOCK:
1.  Title Specificity:
    - The title must describe the *single most unique operation, transformation, formula, or example* introduced in THIS segment.
    - It must reference a *distinct action or detail* that appears ONLY in this portion of the transcript.
    - It must NOT reuse wording from earlier segments unless the transcript itself repeats that wording.
    - It must NOT be a generic topic name (e.g., no "Understanding X", "Overview of Y", or restating the main subject of the lecture).
    - The title must describe the unique action, mechanism, or technical component introduced in this segment.
    - The title must include at least one exact phrase or symbol that appears in the segment and must not rely on generic conceptual wording.

2.  Summary Content:
    - The summary must contain and explain the most complex technical term (if present) from this segment.

3.  Strict Format:
    - The entire output must be a single block of text.
    - The summary must be a single paragraph containing exactly 2–4 sentences.
    - The 'Bullets' section must contain exactly 3 concise bullet points, each starting with '-'.
    - The 'Keywords' section must be a single, comma-separated string containing 5 to 8 technical keywords.

4.  Additional Constraints to Enforce Title Specificity:
    - The title must include at least one concrete noun or phrase taken directly from the transcript (e.g., a variable name, equation component, step name, or specific numerical example).
    - The title must highlight what *changes*, *is computed*, or *is derived* in this segment — not what the general technique is.
    - If the transcript describes an example calculation, formula decomposition, or step-by-step procedure, the title must reference that example or step.

OUTPUT STRUCTURE (EXACTLY AS SHOWN):
Title: <specific, unique chapter title focusing on the distinct action/mechanism in this segment>

Summary: <single paragraph, 2–4 clear, accurate sentences of the main ideas. Do not use conversational filler.>

Bullets:
- <concise bullet point 1>
- <concise bullet point 2>
- <concise bullet point 3>

Keywords: <keyword1, keyword2, keyword3, keyword4, keyword5>
"""

