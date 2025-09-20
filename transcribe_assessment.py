#!/usr/bin/env python3
"""
Assessment Audio Transcription and Analysis Tool
Transcribes MOV files and compares to assessment questions
"""

import whisper
import os
import json
from pathlib import Path

def transcribe_audio_file(file_path, model_name="base"):
    """Transcribe an audio file using Whisper"""
    print(f"Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name)

    print(f"Transcribing: {file_path}")
    result = model.transcribe(str(file_path))

    return {
        "text": result["text"].strip(),
        "language": result["language"],
        "segments": result["segments"]
    }

def analyze_response(question, transcribed_text, written_summary):
    """Analyze the quality of the student's response"""

    analysis = {
        "question": question,
        "transcribed_answer": transcribed_text,
        "written_summary": written_summary,
        "assessment": {}
    }

    # Basic analysis criteria
    word_count = len(transcribed_text.split())
    analysis["assessment"]["word_count"] = word_count
    analysis["assessment"]["has_substantial_content"] = word_count >= 10

    # Check if response addresses the question
    question_keywords = extract_keywords(question)
    response_lower = transcribed_text.lower()
    keyword_matches = sum(1 for keyword in question_keywords if keyword in response_lower)
    analysis["assessment"]["keyword_relevance"] = keyword_matches / len(question_keywords) if question_keywords else 0

    # Compare transcription to written summary
    if written_summary:
        similarity_score = basic_similarity(transcribed_text.lower(), written_summary.lower())
        analysis["assessment"]["transcription_summary_match"] = similarity_score

    return analysis

def extract_keywords(question):
    """Extract key terms from the question"""
    # Simple keyword extraction - remove common words
    stop_words = {"what", "is", "the", "of", "for", "a", "an", "in", "on", "at", "to", "where", "how", "why"}
    words = question.lower().replace("?", "").split()
    return [word for word in words if word not in stop_words and len(word) > 2]

def basic_similarity(text1, text2):
    """Basic similarity score between two texts"""
    words1 = set(text1.split())
    words2 = set(text2.split())

    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0

    intersection = words1.intersection(words2)
    union = words1.union(words2)

    return len(intersection) / len(union)

def main():
    """Main transcription and analysis workflow"""

    # Assessment questions and expected files
    assessment_data = {
        "Q1": {
            "question": "What is the purpose of determining and documenting requirements for a cabinet installation?",
            "file": "IMG_1060.mov",
            "written_summary": "To have a better understanding of the project for all persons involved and plan in advance for any difficulties and refer to documents as a guide for future projects."
        },
        "Q2": {
            "question": "Where on an architectural drawing is the materials list for the project?",
            "file": "77914809189__571E73A4-D2E8-4B00-934C-5B2E54DE47A4.MOV",
            "written_summary": "It's part of the title block or in a separate block usually called the Schedule."
        },
        "Q3": {
            "question": "What information is found in the appliance manuals?",
            "file": "IMG_1062.mov",
            "written_summary": "Fitting instructions and requirements"
        }
    }

    results = {}

    for question_id, data in assessment_data.items():
        file_path = Path(data["file"])

        if file_path.exists():
            print(f"\n=== Processing {question_id} ===")

            try:
                # Transcribe the audio
                transcription = transcribe_audio_file(file_path)

                # Analyze the response
                analysis = analyze_response(
                    data["question"],
                    transcription["text"],
                    data["written_summary"]
                )

                analysis["transcription_details"] = transcription
                results[question_id] = analysis

                print(f"Question: {data['question']}")
                print(f"Transcribed Answer: {transcription['text']}")
                print(f"Written Summary: {data['written_summary']}")
                print(f"Word Count: {analysis['assessment']['word_count']}")
                print(f"Keyword Relevance: {analysis['assessment']['keyword_relevance']:.2f}")

                if 'transcription_summary_match' in analysis['assessment']:
                    print(f"Transcription-Summary Match: {analysis['assessment']['transcription_summary_match']:.2f}")

            except Exception as e:
                print(f"Error processing {question_id}: {e}")
                results[question_id] = {"error": str(e)}
        else:
            print(f"File not found: {file_path}")
            results[question_id] = {"error": f"File not found: {file_path}"}

    # Save results
    with open("assessment_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to assessment_analysis.json")
    return results

if __name__ == "__main__":
    main()