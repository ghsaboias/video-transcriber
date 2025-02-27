"""This is the main entry point of the application."""

import sys
import os
from dotenv import load_dotenv
from utils import (
    create_output_directories,
    generate_output_filename,
    save_outputs,
    generate_output_filename_from_path,
)
from user_interface import (
    select_provider_and_model,
    select_whisper_model,
    select_transcript,
)
from audio_downloader import download_audio
from transcriber import transcribe_audio, check_whisper_setup, setup_whisper
from text_processor import clean_transcript, summarize_text, summarize_from_file

load_dotenv()


def main():
    try:
        # Create output directories
        create_output_directories()

        # Main menu
        print("\nWelcome to Video Transcriber and Summarizer!")
        print("What would you like to do?")
        print("1. Process a YouTube video")
        print("2. Summarize an existing transcript")

        while True:
            choice = input("\nEnter your choice (1-2): ")
            if choice in ["1", "2"]:
                break
            print("Invalid choice. Please enter 1 or 2.")

        # Prompt for provider/model selections for summarization
        print("\nSelect configuration for summarization:")
        provider_summary, model_summary = select_provider_and_model("summarization")

        if choice == "1":
            # Process YouTube URL
            while True:
                url = input("\nEnter YouTube URL (or 'q' to quit): ")
                if url.lower() == "q":
                    return
                if url.strip():
                    break
                print("Please enter a valid URL.")

            # Select Whisper model and language option
            model_name, english_only = select_whisper_model()

            # Check and setup whisper.cpp
            if not check_whisper_setup(model_name, english_only):
                print("Whisper.cpp setup incomplete. Running setup...")
                setup_whisper(model_name, english_only)

            base_filename = generate_output_filename(url)
            audio_path = "temp_audio.wav"

            try:
                # Download and process audio
                print("Downloading audio...")
                audio_path = download_audio(url)
                print(f"Audio downloaded to: {audio_path}")

                if os.path.exists(audio_path):
                    size_mb = os.path.getsize(audio_path) / 1024 / 1024
                    print(f"Audio file size: {size_mb:.2f} MB")

                # Transcribe audio with selected model
                transcript = transcribe_audio(audio_path, model_name, english_only)

                if not transcript:
                    print("Failed to get transcription")
                    return

                # Clean up the transcript
                cleaned_transcript = clean_transcript(transcript)

                # Generate summary
                summary = summarize_text(
                    cleaned_transcript, provider_summary, model_summary
                )

                # Save outputs
                transcript_path, summary_path = save_outputs(
                    cleaned_transcript, summary, base_filename
                )

                print(f"\nTranscription saved to {transcript_path}")
                print(f"Summary saved to {summary_path}")

            finally:
                # Clean up temporary audio file
                if os.path.exists(audio_path):
                    os.remove(audio_path)

        else:  # choice == '2'
            # Select and process existing transcript
            transcript_path = select_transcript()
            if not transcript_path:
                print("No transcript selected. Exiting.")
                return

            base_filename = generate_output_filename_from_path(transcript_path)

            # Generate summary from transcript
            summary = summarize_from_file(
                transcript_path, provider_summary, model_summary
            )

            if summary:
                # Read original transcript for saving
                with open(transcript_path, "r", encoding="utf-8") as f:
                    original_transcript = f.read()

                # Save outputs
                saved_transcript_path, summary_path = save_outputs(
                    original_transcript, summary, base_filename
                )

                print(f"\nTranscription saved to {saved_transcript_path}")
                print(f"Summary saved to {summary_path}")
            else:
                print("Failed to generate summary from transcript")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
