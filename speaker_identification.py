import re
from openai import OpenAI
import os
import nltk

nltk.data.path.append(r"C:\Users\DigitalBluze\nltk_data")
from nltk.tokenize import PunktSentenceTokenizer

tokenizer = PunktSentenceTokenizer()
# text = "Hello. This is a test."
# print(tokenizer.tokenize(text))
from nltk.tokenize import sent_tokenize

# Initialize OpenAI client once
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def split_into_segments(text):
    """
    Split transcript text into list of (timestamp, content).
    """
    pattern = r"(\d{1,2}:\d{2})"
    parts = re.split(pattern, text)
    segments = []

    i = 1
    while i < len(parts) - 1:
        timestamp = parts[i].strip()
        content = parts[i + 1].strip()
        segments.append((timestamp, content))
        i += 2

    return segments


def split_segment_into_sentences(segments):
    """
    Given segments of (timestamp, text), split each text into sentences.
    Return list of (timestamp, sentence).
    """
    sentence_segments = []
    for timestamp, text in segments:
        sentences = sent_tokenize(text)
        for sentence in sentences:
            sentence_segments.append((timestamp, sentence))
    return sentence_segments


def identify_speakers_with_llm(sentence_segments):
    """
    Labels speakers for each sentence segment using LLM.
    Input is a list of (timestamp, sentence).
    Output is labeled transcript as a string, one line per sentence:
    "<timestamp> Speaker N: <sentence>"
    """
    transcript_block = "\n".join([f"{ts} {sentence}" for ts, sentence in sentence_segments])

    prompt = f"""
Assign speaker labels to each timestamped sentence.

Rules:
1. Do NOT rewrite, paraphrase, or modify the text. Only add speaker labels.
2. Keep the original timestamp exactly as provided.
3.Use conversational logic:
    a.Questions are often answered by the other speaker.
    b.Explanations, status updates, or multi-sentence blocks often remain with the same speaker.

4.Name handling rule:
    a.If a sentence directly addresses someone by name
      (e.g., “Yes, Harshit”),
    b.the speaker is most likely the other person — unless that contradicts prior context.
5. Use ONLY these labels: Speaker 1, Speaker 2, or Unknown.
6. If you cannot determine the speaker with high confidence, use "Unknown".
7. Do NOT assign a speaker based on names mentioned inside the sentence.
8. Maintain the original order.

For each line, output exactly:
<timestamp> Speaker N: <sentence>

Example:
0:00 Speaker 1: Hello team, let's start.

Transcript:
{transcript_block}

Return only the labeled transcript lines adhering EXACTLY to the above format.
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=2000,
    )


    return response.choices[0].message.content.strip()


def normalize_llm_output(text):
    """
    Normalize speaker labels capitalization and whitespace.
    """
    text = re.sub(r"\bspeaker\s*1\b", "Speaker 1", text, flags=re.I)
    text = re.sub(r"\bspeaker\s*2\b", "Speaker 2", text, flags=re.I)
    text = re.sub(r"\bunknown\b", "Unknown", text, flags=re.I)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def validate_speaker_labels(text):
    """
    Basic validation of output.
    """
    speakers = set(re.findall(r"(Speaker \d|Unknown)", text))
    timestamps = re.findall(r"\d{1,2}:\d{2}", text)
    errors = []

    if len(speakers) > 3:
        errors.append("Too many speakers detected — possible hallucination.")

    if not timestamps:
        errors.append("No timestamps found — formatting may be corrupted.")

    for s in speakers:
        if s not in {"Speaker 1", "Speaker 2", "Unknown"}:
            errors.append(f"Invalid speaker label found: {s}")

    return errors


# OPTIONAL: Post-processing for name-address correction
def correct_name_address_labels(labeled_output, name_to_speaker):
    corrected_lines = []
    for line in labeled_output.splitlines():
        match = re.match(r'(\d{1,2}:\d{2})\sSpeaker\s(\d):\s(.+)', line)
        if match:
            timestamp, speaker_id, text = match.groups()
            for name, other_speaker_id in name_to_speaker.items():
                if re.search(rf'\b{name}\b', text, re.I):
                    speaker_id = other_speaker_id  # If addressing a name, assign to other speaker
            new_line = f"{timestamp} Speaker {speaker_id}: {text}"
            corrected_lines.append(new_line)
        else:
            corrected_lines.append(line)
    return "\n".join(corrected_lines)


def assign_speakers(full_text, batch_size=12, name_to_speaker=None):
    """
    Full pipeline:
    - Split transcript by timestamps
    - Split each timestamp block into sentences
    - Batch sentences and feed to LLM for speaker labelling
    - Normalize & validate output
    """
    segments = split_into_segments(full_text)
    sentence_segments = split_segment_into_sentences(segments)
    labeled_output = ""
    for i in range(0, len(sentence_segments), batch_size):
        batch = sentence_segments[i:i + batch_size]
        labeled_batch = identify_speakers_with_llm(batch)
        normalized = normalize_llm_output(labeled_batch)
        labeled_output += normalized + "\n"

    # Apply name-address correction if mapping is provided 
    if name_to_speaker:
        labeled_output = correct_name_address_labels(labeled_output, name_to_speaker)

    issues = validate_speaker_labels(labeled_output)
    if issues:
        print("⚠ Speaker labeling warnings:")
        for issue in issues:
            print("-", issue)

    # Flag ambiguous assignments
    unknown_lines = [line for line in labeled_output.splitlines() if "Unknown" in line]
    if unknown_lines:
        print("Review these ambiguous speaker assignments:")
        for line in unknown_lines:
            print(line)

    return labeled_output.strip()



































############################################### ACCURACY 85% ###################################################


# import re
# from openai import OpenAI
# import os
# import nltk
# nltk.data.path.append(r"C:\Users\DigitalBluze\nltk_data")
# from nltk.tokenize import PunktSentenceTokenizer
# tokenizer = PunktSentenceTokenizer()
# # text = "Hello. This is a test."
# # print(tokenizer.tokenize(text))
# from nltk.tokenize import sent_tokenize

# # Initialize OpenAI client once
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# def split_into_segments(text):
#     """
#     Split transcript text into list of (timestamp, content).
#     """
#     pattern = r"(\d{1,2}:\d{2})"
#     parts = re.split(pattern, text)
#     segments = []

#     i = 1
#     while i < len(parts) - 1:
#         timestamp = parts[i].strip()
#         content = parts[i + 1].strip()
#         segments.append((timestamp, content))
#         i += 2

#     return segments


# def split_segment_into_sentences(segments):
#     """
#     Given segments of (timestamp, text), split each text into sentences.
#     Return list of (timestamp, sentence).
#     """
#     sentence_segments = []
#     for timestamp, text in segments:
#         sentences = sent_tokenize(text)
#         for sentence in sentences:
#             sentence_segments.append((timestamp, sentence))
#     return sentence_segments


# def identify_speakers_with_llm(sentence_segments):
#     """
#     Labels speakers for each sentence segment using LLM.
#     Input is a list of (timestamp, sentence).
#     Output is labeled transcript as a string, one line per sentence:
#     "<timestamp> Speaker N: <sentence>"
#     """
#     transcript_block = "\n".join([f"{ts} {sentence}" for ts, sentence in sentence_segments])

#     prompt = f"""
# Assign speaker labels to each timestamped sentence.

# Rules:
# 1. Do NOT rewrite, paraphrase, or modify the text. Only add speaker labels.
# 2. Keep the original timestamp exactly as provided.
# 3. Use ONLY these labels: Speaker 1, Speaker 2, or Unknown.
# 4. If you cannot determine the speaker with high confidence, use "Unknown".
# 5. Do NOT assign a speaker based on names mentioned inside the sentence.
# 6. Maintain the original order.

# For each line, output exactly:
# <timestamp> Speaker N: <sentence>

# Example:
# 0:00 Speaker 1: Hello team, let's start.

# Transcript:
# {transcript_block}

# Return only the labeled transcript lines adhering EXACTLY to the above format.
# """

#     response = client.chat.completions.create(
#         model="gpt-4",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0,
#         max_tokens=2000,
#     )

#     return response.choices[0].message.content.strip()


# def normalize_llm_output(text):
#     """
#     Normalize speaker labels capitalization and whitespace.
#     """
#     text = re.sub(r"\bspeaker\s*1\b", "Speaker 1", text, flags=re.I)
#     text = re.sub(r"\bspeaker\s*2\b", "Speaker 2", text, flags=re.I)
#     text = re.sub(r"\bunknown\b", "Unknown", text, flags=re.I)

#     lines = [line.strip() for line in text.splitlines() if line.strip()]
#     return "\n".join(lines)


# def validate_speaker_labels(text):
#     """
#     Basic validation of output.
#     """
#     speakers = set(re.findall(r"(Speaker \d|Unknown)", text))
#     timestamps = re.findall(r"\d{1,2}:\d{2}", text)
#     errors = []

#     if len(speakers) > 3:
#         errors.append("Too many speakers detected — possible hallucination.")

#     if not timestamps:
#         errors.append("No timestamps found — formatting may be corrupted.")

#     for s in speakers:
#         if s not in {"Speaker 1", "Speaker 2", "Unknown"}:
#             errors.append(f"Invalid speaker label found: {s}")

#     return errors


# def assign_speakers(full_text, batch_size=12):
#     """
#     Full pipeline:
#     - Split transcript by timestamps
#     - Split each timestamp block into sentences
#     - Batch sentences and feed to LLM for speaker labelling
#     - Normalize & validate output
#     """
#     segments = split_into_segments(full_text)
#     sentence_segments = split_segment_into_sentences(segments)

#     labeled_output = ""
#     for i in range(0, len(sentence_segments), batch_size):
#         batch = sentence_segments[i:i + batch_size]
#         labeled_batch = identify_speakers_with_llm(batch)
#         normalized = normalize_llm_output(labeled_batch)
#         labeled_output += normalized + "\n"

#     issues = validate_speaker_labels(labeled_output)
#     if issues:
#         print("⚠ Speaker labeling warnings:")
#         for issue in issues:
#             print("-", issue)

#     return labeled_output.strip()





















######################################################## LOW ACCURACY #######################################################


# # --------------------------------------------------------------
# # With Timestamp
# # --------------------------------------------------------------

# import re 
# import openai


# # --------------------------------------------------------------
# # Utility: Split transcript by timestamp blocks
# # --------------------------------------------------------------

# def split_into_segments(text):
#     """
#     Splits the transcript into segments based on timestamps
#     Example:
#         0:00 Lorem ipsum...
#         0:12 Another part...
#     Returns a list of (timestamp, text).
#     """
#     pattern = r"(\d{1,2}:\d{2})"
#     parts = re.split(pattern,text)

#     segments = []
#     for i in range(1, len(parts),2):
#         timestamp = parts[i]
#         content = parts[i+1].strip()
#         segments.append((timestamp,content))

#     return segments


# # --------------------------------------------------------------
# # Core LLM call for speaker identification
# # --------------------------------------------------------------

# def identify_speakers_with_llm(segments):
#     """
#     Sends a small batch of segments to the LLM and returns the output.
#     Uses strict instructions to:
#     - prevent rewriting
#     - avoid hallucination
#     - maintain timestamp structure
#     """
#     transcript_block = ""
#     for ts,content in segments:
#         transcript_block+=f"{ts} {content}\n"

#     prompt = f"""
# Assign speaker labels to each timestamp entry.

# Rules:
# 1. Do NOT rewrite, paraphrase, or modify the text. Only add speaker labels.
# 2. Keep the original timestamp exactly as provided.
# 3. Use ONLY these labels: Speaker 1, Speaker 2, or Unknown.
# 4. If you cannot determine the speaker with high confidence, use "Unknown".
# 5. If a name is mentioned in the line (e.g., "Thanks, Prabhat"), the speaker is NOT that person.
# 6. Maintain the same sequence of lines; do not merge or rearrange anything.

# Transcript:
# {transcript_block}

# Return the labeled transcript in the same order.
# """
#     response = openai.chat.completions.create(
#         model= "gpt-4",
#         messages= [{"role": "user","content":prompt}],
#         temperature=0,
#         max_tokens=1200,
#     )

#     return response.choices[0].message.content.strip()

# # --------------------------------------------------------------
# # Cleanup LLM output for consistency
# # --------------------------------------------------------------

# def normalize_llm_output(text):
#     """
#     Ensures labels are consistent and formatting is correct.
#     """
#     # Normalize capitalization
#     text = re.sub(r"(speaker\s*1)", "Speaker 1", text, flags=re.I)
#     text = re.sub(r"(speaker\s*2)", "Speaker 2", text, flags=re.I)
#     text = re.sub(r"(unknown)", "Unknown", text, flags=re.I)

#     # Ensure timestamp and label order consistency
#     text = re.sub(r"(\d{1,2}:\d{2})\s*(Speaker)", r"\1\n\2", text)

#     return text


# # --------------------------------------------------------------
# # Validate labels (catch hallucination cases)
# # --------------------------------------------------------------

# def validate_speaker_labels(text):
#     """
#     Detects potential hallucinated patterns such as:
#     - too many speakers
#     - inconsistent naming
#     - missing timestamps
#     """
#     speakers = set(re.findall(r"(Speaker \d|Unknown)",text))
#     timestamps = re.findall(r"\d{1,2}:\d{2}", text)

#     errors = []

#     if len(speakers)>3:
#         errors.append("Too many speakers detected — possible hallucination.")

#     if not timestamps:
#         errors.append("No timestamps found — formatting may be corrupted.")

#     for s in speakers:
#         if not re.match(r"Speaker [12]|Unknown", s):
#             errors.append(f"Invalid speaker label found: {s}")

#     return errors

# # --------------------------------------------------------------
# # Main wrapper function used by pipeline
# # --------------------------------------------------------------

# def assign_speakers(full_text):
#     """
#     End-to-end speaker identification:
#     - split transcript
#     - batch segments in groups
#     - run LLM
#     - normalize output
#     - validate output
#     """
#     segments = split_into_segments(full_text)
#     # Batch size (3–5 segments at a time)
#     BATCH = 4

#     labeled_output = ""

#     for i in range(0, len(segments), BATCH):
#         chunk = segments[i:i + BATCH]
#         labeled_chunk = identify_speakers_with_llm(chunk)
#         normalized = normalize_llm_output(labeled_chunk)
#         labeled_output += normalized + "\n\n"

#     # Validation
#     issues = validate_speaker_labels(labeled_output)
#     if issues:
#         print("⚠ Speaker labeling warnings:\n", issues)

#     return labeled_output.strip()