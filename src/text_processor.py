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
    text: str,
    provider: str = "openrouter",
    model: Optional[str] = None,
    use_english: bool = True,
) -> str:
    """
    Summarize text using the selected provider and model.
    If use_english is False, the summary will be in the same language as the input text.
    """
    language_instruction = (
        ""
        if use_english
        else ", keep the summary in the same language of the input text (do NOT translate to English!)"
    )
    prompt = f"""Summarize the provided transcript in a professional tone{language_instruction}. Structure the summary in 3 sections with the following purposes, using bolded headings:
    1. A concise statement of the content’s primary focus, purpose, or subject matter, avoiding specific details or data.
    2. An outline of the most significant facts, events, or elements presented, including relevant specifics such as individuals, locations, or figures, in precise and objective language.
    3. A highlight of the content’s broader impact, practical utility, or significance, including potential effects, insights, or value to an audience as applicable; if none are evident, a brief summary of the central theme or purpose.

    Maintain a formal and concise style, focusing solely on the content. Do NOT reference the transcript as a source, and do NOT include introductory text like 'Here is the transcript:' or 'Here is the summary:', provide just the summary. Transcript:
    {text}"""

    print(f"Starting summarization with {provider} (model: {model or 'default'})...")

    if provider.lower() == "openrouter":
        if model is None:
            model = "anthropic/claude-3-sonnet-20240229"
        try:
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable is not set")

            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
                default_headers={
                    "HTTP-Referer": "https://github.com/ghsaboias/video-transcriber",
                    "X-Title": "Video Transcriber",
                },
            )
            print("Sending request to OpenRouter API...")
            print(f"Using model: {model}")

            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8000,
            )

            if not response:
                raise ValueError("Received empty response from OpenRouter API")

            if not hasattr(response, "choices") or not response.choices:
                raise ValueError(
                    f"Invalid response format from OpenRouter API: {response}"
                )

            summary = response.choices[0].message.content.strip()
            print("Summarization completed successfully.")
        except ValueError as ve:
            print(f"Configuration error: {str(ve)}")
            summary = "Failed to generate summary: Configuration error"
        except Exception as e:
            print(f"Error during summarization with OpenRouter: {str(e)}")
            print(
                f"Response received: {response if 'response' in locals() else 'No response'}"
            )
            summary = "Failed to generate summary: API error"
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
    file_path: str,
    provider: str = "openrouter",
    model: Optional[str] = None,
    use_english: bool = True,
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
        summary = summarize_text(cleaned_transcript, provider, model, use_english)

        return summary
    except Exception as e:
        print(f"Error processing transcript file: {str(e)}")
        return None
