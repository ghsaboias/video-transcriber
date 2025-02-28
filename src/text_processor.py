"""This module contains functions for processing text, including cleaning transcripts and generating detailed overviews."""

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


def strip_introductory_text(overview: str) -> str:
    """
    Strip introductory text from overview if the first line doesn't end with "**".
    This removes common phrases like "Here's a structured overview..." that models sometimes include.
    """
    lines = overview.split("\n")

    # If there's at least one line and it doesn't end with "**" (markdown bold)
    if lines and not lines[0].strip().endswith("**"):
        # Remove the first line
        return "\n".join(lines[1:]).strip()

    return overview


def generate_detailed_overview(
    text: str,
    provider: str = "openrouter",
    model: Optional[str] = None,
    use_english: bool = True,
) -> str:
    """
    Generate a detailed overview of text using the selected provider and model.
    If use_english is False, the overview will be in the same language as the input text.
    Metadata (channel name and video title) is extracted from the transcript if available.
    """
    language_instruction = (
        ""
        if use_english
        else ", keep the overview in the same language of the input text (do NOT translate to English!)"
    )

    prompt = f"""Generate a detailed overview of the provided text in a professional tone{language_instruction}. Structure the overview in 3 sections with the following purposes, using bolded headings:
    1. A concise statement of the content's core focus or thesis, capturing its primary intent or subject matter without delving into specifics.
    2. A comprehensive outline of key insights, facts, events, predictions, or takeaways, blending concrete details (e.g., current developments, specific examples) and speculative elements (e.g., future implications, forecasts) as present. Include relevant specifics such as individuals, technologies, or figures, using subheadings or bullet points for clarity and emphasis on actionable or noteworthy points.
    3. A summary of the content's stated broader implications, relevance, or practical utility as presented, focusing on its significance for decision-making, societal impact, or intellectual value across relevant domains (e.g., technology, geopolitics), reflecting only the perspectives or conclusions offered in the text.

    Scale the overview's length proportionally to the input, targeting 300-600 words unless the content warrants more.

    Output just the overview, do NOT reference the text as a source, do NOT include 'Here is a structured overview of the text:'. Text:

    {text}"""

    print(
        f"Starting detailed overview generation with {provider} (model: {model or 'default'})..."
    )

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

            overview = response.choices[0].message.content.strip()
        except ValueError as ve:
            print(f"Configuration error: {str(ve)}")
            overview = "Failed to generate detailed overview: Configuration error"
        except Exception as e:
            print(
                f"Error during detailed overview generation with OpenRouter: {str(e)}"
            )
            print(
                f"Response received: {response if 'response' in locals() else 'No response'}"
            )
            overview = "Failed to generate detailed overview: API error"
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
            overview = response.choices[0].message.content.strip()
            if "deepseek" in model.lower():
                overview = re.sub(
                    r"<\s*think\s*>.*?<\s*/\s*think\s*>", "", overview, flags=re.DOTALL
                ).strip()
            print("Detailed overview generation completed.")
        except Exception as e:
            print(f"Error during detailed overview generation with Groq: {str(e)}")
            overview = "Failed to generate detailed overview."
    else:
        print("Unknown provider specified.")
        overview = "Failed to generate detailed overview."

    # Strip any introductory text before returning
    return strip_introductory_text(overview)


def generate_overview_from_file(
    file_path: str,
    provider: str = "openrouter",
    model: Optional[str] = None,
    use_english: bool = True,
) -> str:
    """
    Read a transcript from a file and generate a detailed overview
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            transcript = f.read()

        # Clean the transcript
        cleaned_transcript = clean_transcript(transcript)

        # Generate detailed overview
        overview = generate_detailed_overview(
            cleaned_transcript, provider, model, use_english
        )

        return overview
    except Exception as e:
        print(f"Error processing transcript file: {str(e)}")
        return None


# For backward compatibility
summarize_text = generate_detailed_overview
summarize_from_file = generate_overview_from_file
