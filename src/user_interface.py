"""This module contains functions for user interaction and selection."""

from src.config import PROVIDERS_AND_MODELS, WHISPER_MODELS
from src.utils import list_available_transcripts
from typing import Tuple, Optional


def select_content_source() -> Tuple[str, bool]:
    """
    Ask user to select content source and language.
    Returns tuple of (source_type, is_english) where:
    - source_type is "youtube" or "transcript"
    - is_english is True if content is in English, False otherwise
    """
    print("\nWelcome to Video Transcriber and Summarizer!")
    print("What would you like to do?")
    print("1. Process a YouTube video")
    print("2. Summarize an existing transcript")

    while True:
        choice = input("\nEnter your choice (1-2): ")
        if choice in ["1", "2"]:
            break
        print("Invalid choice. Please enter 1 or 2.")

    source_type = "youtube" if choice == "1" else "transcript"

    # Ask about language for both YouTube videos and transcripts
    while True:
        content_type = "video" if source_type == "youtube" else "transcript"
        lang_choice = input(
            f"\nIs your {content_type} in English?\n1. Yes\n2. No\nEnter your choice: "
        ).lower()
        if lang_choice in ["1", "2"]:
            is_english = lang_choice == "1"
            break
        print("Invalid choice. Please enter '1' or '2'.")

    return source_type, is_english


def select_provider_and_model(role: str) -> Tuple[str, str]:
    """
    Prompt the user to select a provider and a model for the given role.
    Role can be 'summarization' or 'summary review'.
    Returns a tuple of (provider, model).
    """
    print(f"\nSelect provider for {role}:")
    providers = list(PROVIDERS_AND_MODELS.keys())
    for idx, provider_key in enumerate(providers, start=1):
        print(f"{idx}. {PROVIDERS_AND_MODELS[provider_key]['display']}")
    while True:
        choice = input("Enter your choice number: ")
        if choice.isdigit() and 1 <= int(choice) <= len(providers):
            selected_provider = providers[int(choice) - 1]
            break
        else:
            print("Invalid input. Please enter a valid number.")

    models = PROVIDERS_AND_MODELS[selected_provider]["summary_models"]
    print(
        f"\nSelect model for {role} using {PROVIDERS_AND_MODELS[selected_provider]['display']}:"
    )
    for idx, (model_display, model_keyword) in enumerate(models, start=1):
        print(f"{idx}. {model_display} [{model_keyword}]")
    while True:
        choice_model = input("Enter your choice number: ")
        if choice_model.isdigit() and 1 <= int(choice_model) <= len(models):
            selected_model = models[int(choice_model) - 1][1]
            break
        else:
            print("Invalid input. Please enter a valid number.")

    return selected_provider, selected_model


def select_whisper_model(is_english_content: bool = False) -> Tuple[str, bool]:
    """
    Returns the small Whisper model name and language mode, automatically selecting between
    English-only (faster) for English content and multilingual for other languages.
    Returns tuple of (model_name, is_english_only)
    """
    selected_model = "small"
    is_english_only = is_english_content

    print("\nUsing Whisper small model:")
    print(f"Model size: {WHISPER_MODELS[selected_model]['size_mb']}MB")
    print(f"Description: {WHISPER_MODELS[selected_model]['description']}")

    if is_english_content:
        print(
            "Using English-only version (faster and more accurate for English content)"
        )
    else:
        print("Using multilingual version")
        print(
            f"Supported languages: {WHISPER_MODELS[selected_model]['multilingual']['languages']}"
        )

    return selected_model, is_english_only


def select_transcript() -> Optional[str]:
    """
    Display available transcripts and let user select one
    Returns the path to the selected transcript or None if no selection made
    """
    transcripts = list_available_transcripts()

    if not transcripts:
        print("No transcripts found in outputs/transcripts directory.")
        return None

    print("\nAvailable transcripts:")
    for idx, transcript in enumerate(transcripts, start=1):
        print(f"{idx}. {transcript.name}")

    while True:
        choice = input(
            "\nEnter the number of the transcript to summarize (or 'q' to quit): "
        )
        if choice.lower() == "q":
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(transcripts):
            return str(transcripts[int(choice) - 1])
        print("Invalid selection. Please try again.")


def select_summary_language(
    is_english_model: bool, is_english_content: bool = False
) -> bool:
    """
    Determine the language for the summary.
    Returns True for English, False for original language.
    """
    if is_english_model:
        print(
            "\nSince you selected an English-only model, the summary will be in English."
        )
        return True

    if is_english_content:
        print("\nSince the content is in English, the summary will be in English.")
        return True

    # Only ask for non-English content with multilingual models
    while True:
        choice = input(
            "\nWould you like the summary in English or the original language?\n1. English\n2. Original language\nEnter your choice: "
        ).lower()
        if choice in ["1", "2"]:
            return choice == "1"
        print(
            "Invalid choice. Please enter '1' for English or '2' for original language."
        )
