"""This module contains functions for user interaction and selection."""

from config import PROVIDERS_AND_MODELS, WHISPER_MODELS
from utils import list_available_transcripts
from typing import Tuple, Optional


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


def select_whisper_model() -> Tuple[str, bool]:
    """
    Let user select a Whisper model and whether to use English-only version
    Returns tuple of (model_name, is_english_only)
    """
    print("\nSelect Whisper model for transcription:")
    models = list(WHISPER_MODELS.keys())
    for idx, model in enumerate(models, start=1):
        info = WHISPER_MODELS[model]
        print(f"\n{idx}. {model} ({info['size_mb']}MB)")
        print(f"   Description: {info['description']}")
        print(f"   Language Support:")
        if info["english_only"] is not None:
            print(
                f"   - English-only version available (faster and more accurate for English)"
            )
        if info["multilingual"]["languages"]:
            print(
                f"   - Multilingual version supports: {info['multilingual']['languages']}"
            )

    while True:
        choice = input("\nEnter model number: ")
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            selected_model = models[int(choice) - 1]
            break
        print("Invalid choice. Please try again.")

    if WHISPER_MODELS[selected_model]["english_only"] is not None:
        print("\nThis model supports both English-only and multilingual versions.")
        print("- English-only: Faster and more accurate for English content")
        print(
            f"- Multilingual: Supports {WHISPER_MODELS[selected_model]['multilingual']['languages']}"
        )
        while True:
            lang_choice = input(
                "\nDo you want to use English-only version? [y/n]: "
            ).lower()
            if lang_choice in ["y", "n"]:
                break
            print("Invalid choice. Please enter 'y' or 'n'.")

        is_english = lang_choice == "y"
    else:
        print(
            f"\nNote: This model only supports multilingual version with: {WHISPER_MODELS[selected_model]['multilingual']['languages']}"
        )
        is_english = False

    if not is_english:
        print(
            "\nMultilingual model selected. This will work with any of the supported languages automatically."
        )
        print(
            "Just provide the video URL, and the model will detect and transcribe the language."
        )

    return selected_model, is_english


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
