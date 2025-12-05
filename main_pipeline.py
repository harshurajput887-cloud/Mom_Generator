from clean_transcript import *
from speaker_identification import assign_speakers
from mom_generator.mom_extraction import (
    extract_action_items,
    extract_decisions,
    extract_questions,
    summarize_discussion,
)
from mom_generator.mom_formatter import format_mom_html, format_mom_pdf

RAW_TRANSCRIPT = "data/transcript.txt"
CLEANED_TRANSCRIPT = "data/cleaned_transcript.txt"
SPEAKER_LABELED_TRANSCRIPT = "data/speaker_labeled_transcript.txt"
FILLER_JSON_PATH = "data/filler_words.json"
FINAL_MOM_PATH = "data/final_mom.txt"

# Step 1: Clean transcript
filler_words = load_filler_words(FILLER_JSON_PATH)
raw_text = load_transcript(RAW_TRANSCRIPT)
cleaned_text = remove_filler_words_preserving_structure(raw_text, filler_words)
chunks = chunk_text(cleaned_text)
corrected_chunks = [fix_grammar_with_openai(chunk) for chunk in chunks]
final_cleaned_text = "\n".join(corrected_chunks)
save_clean_transcript(final_cleaned_text, CLEANED_TRANSCRIPT)
print(f"Cleaned transcript saved: {CLEANED_TRANSCRIPT}")

# Step 2: Speaker Identification
speaker_labeled_text = assign_speakers(final_cleaned_text)
with open(SPEAKER_LABELED_TRANSCRIPT, "w", encoding="utf-8") as f:
    f.write(speaker_labeled_text)
print(f"Speaker-labeled transcript saved: {SPEAKER_LABELED_TRANSCRIPT}")

# Step 3: MoM Extraction
summary = summarize_discussion(speaker_labeled_text)
actions = extract_action_items(speaker_labeled_text)
decisions = extract_decisions(speaker_labeled_text)
questions = extract_questions(speaker_labeled_text)

mom_lines = []
mom_lines.append("SUMMARY:")
mom_lines.append(summary)
mom_lines.append("\nACTION ITEMS:")
mom_lines.extend(actions)
mom_lines.append("\nDECISIONS:")
mom_lines.extend(decisions)
mom_lines.append("\nQUESTIONS / ISSUES:")
mom_lines.extend(questions)

with open(FINAL_MOM_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(mom_lines))

# Generate professional formats
mom_content = "\n".join(mom_lines)
html_path = format_mom_html(mom_content)
pdf_path = format_mom_pdf(mom_content)

print(f"Pipeline Complete!")
print(f"Text MoM: {FINAL_MOM_PATH}")
print(f"HTML: {html_path} (email-ready)")
print(f"PDF: {pdf_path} (professional)")