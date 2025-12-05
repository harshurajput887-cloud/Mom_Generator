# mom_extraction.py
from typing import List
import os
import re
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# ACTION ITEMS
# -----------------------------

def extract_action_items(transcript: str) -> List[str]:


    """
    Extract ONLY clear action items (tasks, follow-ups, assignments, deadlines)
    from any speaker-labeled transcript.
    Returns a list of bullet-style strings with Speaker 1/2.
    """

    prompt = f"""
You are generating Minutes of Meeting.

From the speaker-labeled transcript below, extract ONLY clear action items:
tasks, follow-ups, assignments, or work with deadlines.

For each action item, include:
- Who is responsible (use the Speaker 1 / Speaker 2 labels exactly as in the text).
- What must be done.
- By when, if a deadline is mentioned.

IMPORTANT:
- Do NOT include high-level decisions or agreements here.
- Do NOT include general discussion or questions.

Transcript:
{transcript}

Return ONLY bullet points in this format:
- [Speaker X] Concrete task description with deadline if available.
"""

    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=700,
    )
    content = resp.choices[0].message.content.strip()
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]

    # Fallback: simple pattern for obvious tasks if LLM returns nothing
    if not lines:
        pattern = r'(Speaker \d: .*?(?:should|will|please|need to|prepare|update|review|send|share|finalize|draft).*)'
        matches = re.findall(pattern, transcript, flags=re.I)
        lines = []
        for m in matches:
            speaker_match = re.search(r'(Speaker \d)', m)
            speaker = speaker_match.group(1) if speaker_match else "Speaker ?"
            sentence = re.sub(r'^Speaker \d:\s*', '', m)
            lines.append(f"- [{speaker}] {sentence}")

    return lines



# -----------------------------
# DECISIONS (no duplicated tasks)
# -----------------------------

def extract_decisions(transcript: str) -> List[str]:
    """
    Extract only high-level decisions / agreements (NOT individual tasks).
    Returns a list of bullet-style strings.
    """

    prompt = f"""
From the transcript below, extract only high-level decisions or agreements,
not individual tasks.

Examples of decisions:
- Agreeing to a plan or approach.
- Confirming timelines at a high level.
- Deciding to focus on a particular area.
- Agreeing on when to meet next.

Do NOT:
- Repeat concrete tasks or assignments already suitable as action items.
- Include generic discussion.

Transcript:
{transcript}

Return ONLY bullet points like:
- Short description of what was agreed.
"""

    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500,
    )
    content = resp.choices[0].message.content.strip()
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]

    return lines


# -----------------------------
# QUESTIONS / ISSUES
# -----------------------------

def extract_questions(transcript: str) -> List[str]:
    """
    Extract open questions or unresolved issues from the transcript.
    Returns a list of bullet-style strings with Speaker labels.
    """

    prompt = f"""
From the transcript below, list the main open questions or issues
raised during the meeting. Include who asked them using Speaker 1/2 labels.

Transcript:
{transcript}

Return ONLY bullet points like:
- [Speaker X] Question text
"""

    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500,
    )
    content = resp.choices[0].message.content.strip()
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]

    # Fallback: any line with a question mark
    if not lines:
        for ln in transcript.splitlines():
            if '?' in ln:
                # try to keep Speaker label
                m = re.match(r'(Speaker \d:)\s*(.*)', ln)
                if m:
                    lines.append(f"- [{m.group(1).replace(':','')}] {m.group(2)}")
                else:
                    lines.append(f"- {ln}")

    return lines


# -----------------------------
# SUMMARY
# -----------------------------

def summarize_discussion(transcript: str) -> str:
    """
    Generate a concise 4–5 sentence summary for any meeting transcript.
    """

    prompt = f"""
Summarize this meeting in 4–5 concise sentences.

Include:
- Overall project/status.
- Any issues or delays.
- Tools/systems mentioned.
- Main focus for the upcoming period.
- Documentation or process updates.
- Next meeting plan, if mentioned.

Use neutral wording and do NOT list bullet points.
Use Speaker 1 / Speaker 2 labels only if needed for clarity.

Transcript:
{transcript}
"""

    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500,
    )
    summary = resp.choices[0].message.content.strip()
    return summary


# -----------------------------
# Simple manual test
# -----------------------------

if __name__ == "__main__":
    with open("data/speaker_labeled_transcript.txt", "r", encoding="utf-8") as f:
        t = f.read()

    print("SUMMARY:\n", summarize_discussion(t), "\n")
    print("ACTION ITEMS:")
    for a in extract_action_items(t):
        print(a)
    print("\nDECISIONS:")
    for d in extract_decisions(t):
        print(d)
    print("\nQUESTIONS / ISSUES:")
    for q in extract_questions(t):
        print(q)
