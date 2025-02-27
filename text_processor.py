"""This module contains functions for processing text, including cleaning transcripts and generating summaries."""

import re
import os
from typing import Optional
from openai import OpenAI
from groq import Groq


def clean_transcript(text: str) -> str:
    """
    Remove timestamps from transcript and clean up the text
    Format: [00:00:00.000 --> 00:00:03.920]   Text
    """
    # Remove timestamp lines and clean up extra whitespace
    cleaned = re.sub(r"\[[0-9:.\s\->\s]+\]\s*", "", text)

    # Remove multiple spaces
    cleaned = re.sub(r"\s+", " ", cleaned)

    # Remove [BLANK_AUDIO] markers
    cleaned = re.sub(r"\[BLANK_AUDIO\]", "", cleaned)

    # Remove extra newlines
    cleaned = re.sub(r"\n+", "\n", cleaned)

    # Strip leading/trailing whitespace
    cleaned = cleaned.strip()

    return cleaned


def summarize_text(
    text: str, provider: str = "openrouter", model: Optional[str] = None
) -> str:
    """
    Summarize text using the selected provider and model.
    """
    prompt = f"""
I'd like you to summarize the following content in a concise yet comprehensive way, capturing all key information without missing valuable details. Organize your response into three sections: (1) Factual Information (core facts, concepts, or claims presented, with brief explanations where needed for clarity), (2) Predictions or Insights (any future implications, trends, or outcomes suggested, with examples or reasoning if provided), and (3) Recommendations or Takeaways (specific, actionable steps, lessons, or suggestions for the audience, if applicable, or notable conclusions). Ensure the summary is clear, structured, and reflects the full scope—facts, ideas, and practical points—while avoiding vague or generic statements. Adapt the depth of each section to the content, prioritizing what's most significant. Here's the content: {text}
"""
    print(f"Starting summarization with {provider} (model: {model or 'default'})...")

    if provider.lower() == "openrouter":
        if model is None:
            model = "anthropic/claude-3-sonnet-20240229"
        try:
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_API_KEY"),
                default_headers={
                    "HTTP-Referer": "https://github.com/yourusername/video-transcriber",  # Replace with your repo URL
                    "X-Title": "Video Transcriber",  # Replace with your app name
                },
            )
            print("Sending request to OpenRouter API...")
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000,
            )
            summary = response.choices[0].message.content.strip()
            print("Summarization completed.")
        except Exception as e:
            print(f"Error during summarization with OpenRouter: {str(e)}")
            summary = "Failed to generate summary."
    elif provider.lower() == "groq":
        if model is None:
            model = "llama-3.3-70b-versatile"
        try:
            groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            print("Sending request to Groq API...")
            response = groq_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "assistant", "content": ""},
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=4000,
            )
            summary = response.choices[0].message.content.strip()
            if "deepseek" in model.lower():
                summary = re.sub(
                    r"<\s*think\s*>.*?<\s*/\s*think\s*>", "", summary, flags=re.DOTALL
                ).strip()
            print("Summarization completed.")
        except Exception as e:
            print(f"Error during summarization with Groq: {str(e)}")
            summary = "Failed to generate summary."
    else:
        print("Unknown summarization provider specified.")
        summary = "Failed to generate summary."

    return summary


def summarize_from_file(
    file_path: str, provider: str = "openrouter", model: Optional[str] = None
) -> str:
    """
    Read a transcript from a file and generate a summary
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            transcript = f.read()

        # Clean the transcript
        cleaned_transcript = clean_transcript(transcript)

        # Generate summary
        summary = summarize_text(cleaned_transcript, provider, model)

        return summary
    except Exception as e:
        print(f"Error processing transcript file: {str(e)}")
        return None
