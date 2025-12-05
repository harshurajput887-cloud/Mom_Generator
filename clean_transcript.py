import os
import re
import json
from dotenv import load_dotenv
from openai import OpenAI

# -------------------------------------------------------
# LOAD API KEY
# -------------------------------------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------------------------------------
# FILE PATHS
# -------------------------------------------------------
FILLER_JSON_PATH = "filler_words.json"
INPUT_TRANSCRIPT_PATH = "transcript.txt"
OUTPUT_TRANSCRIPT_PATH = "cleaned_transcript.txt"

# -------------------------------------------------------
# LOADERS & SAVERS
# -------------------------------------------------------
def load_filler_words(json_filepath):
    with open(json_filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [w.strip() for w in data.get("filler_words", []) if w.strip()]


def load_transcript(txt_filepath):
    with open(txt_filepath, "r", encoding="utf-8") as f:
        return f.read()


def save_clean_transcript(clean_text, output_filepath):
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(clean_text)


# -------------------------------------------------------
# REGEX HELPERS
# -------------------------------------------------------
def build_filler_pattern(filler_words):
    escaped = [re.escape(w) for w in filler_words]
    pattern = r"\b(" + "|".join(escaped) + r")\b"
    return re.compile(pattern, flags=re.IGNORECASE)


# -------------------------------------------------------
# FILLER WORD CLEANING
# -------------------------------------------------------
def remove_filler_words_preserving_structure(text, filler_words):
    """
    Removes filler words but preserves:
    - punctuation
    - timestamps
    - casing
    """

    if not filler_words:
        return text

    # Protect timestamps
    timestamps = re.findall(r"\d+:\d+", text)
    temp_map = {ts: f"__TS_{i}__" for i, ts in enumerate(timestamps)}

    # Replace timestamps with placeholders
    for ts, placeholder in temp_map.items():
        text = text.replace(ts, placeholder)

    # Remove filler words
    filler_pattern = build_filler_pattern(filler_words)
    text = filler_pattern.sub("", text)

    # Clean spacing & punctuation
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"\?\s*,\s*", "? ", text)
    text = re.sub(r"\.\s*,\s*", ". ", text)

    # Capitalize sentences
    text = re.sub(
        r"(^|[.!?]\s+)([a-z])",
        lambda m: m.group(1) + m.group(2).upper(),
        text,
    )

    # Restore timestamps
    for ts, placeholder in temp_map.items():
        text = text.replace(placeholder, ts)

    return text.strip()


# -------------------------------------------------------
# GRAMMAR FIX USING gpt-4o-mini (STRICT MODE)
# -------------------------------------------------------
def fix_grammar_with_openai(text):
    prompt = f"""
Fix ONLY grammar, punctuation, and sentence boundaries in the transcript below.

STRICT RULES:
- Do NOT add quotation marks.
- Do NOT rewrite or rephrase sentences.
- Do NOT add or remove content.
- Do NOT merge unrelated sentences.
- Preserve ALL timestamps exactly (like 0:00).
- Preserve names (e.g., Prabhat, Harshit).
- Preserve technical terms (ETL, Power BI, pipeline).
- Fix ONLY grammar & punctuation — nothing else.

Transcript:
{text}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print("OpenAI Error:", e)
        print("Returning uncorrected text.")
        return text


# -------------------------------------------------------
# CHUNK LONG TEXTS (if needed)
# -------------------------------------------------------
def chunk_text(text, max_chars=9000):
    """Splits text safely for long transcripts."""
    if len(text) <= max_chars:
        return [text]

    parts = []
    while len(text) > max_chars:
        split_idx = text.rfind("\n", 0, max_chars)
        if split_idx == -1:
            split_idx = max_chars
        parts.append(text[:split_idx])
        text = text[split_idx:]

    if text.strip():
        parts.append(text)

    return parts


# -------------------------------------------------------
# MAIN PROCESS
# -------------------------------------------------------
def main():
    filler_words = load_filler_words(FILLER_JSON_PATH)
    raw_text = load_transcript(INPUT_TRANSCRIPT_PATH)

    print("Removing filler words...")
    cleaned_text = remove_filler_words_preserving_structure(raw_text, filler_words)

    print("Splitting transcript for safe processing...")
    chunks = chunk_text(cleaned_text)

    print("Correcting grammar using gpt-4o-mini...")
    corrected_chunks = [fix_grammar_with_openai(chunk) for chunk in chunks]
    final_text = "\n".join(corrected_chunks)

    print("Saving cleaned transcript...")
    save_clean_transcript(final_text, OUTPUT_TRANSCRIPT_PATH)

    print(f"Cleaned transcript saved to '{OUTPUT_TRANSCRIPT_PATH}'")


if __name__ == "__main__":
    main()