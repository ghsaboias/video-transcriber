"""This is the main entry point of the application."""

import sys
import os
from dotenv import load_dotenv
from src.utils import (
    create_output_directories,
    generate_output_filename,
    save_outputs,
    generate_output_filename_from_path,
)
from src.user_interface import (
    select_content_source,
    select_provider_and_model,
    select_whisper_model,
    select_transcript,
    select_summary_language,
)
from src.audio_downloader import download_audio
from src.transcriber import transcribe_audio, check_whisper_setup, setup_whisper
from src.text_processor import clean_transcript, summarize_text, summarize_from_file

load_dotenv()


def main():
    try:
        # Create output directories
        create_output_directories()

        # Get content source and language info
        source_type, is_english_content = select_content_source()

        # Prompt for provider/model selections for summarization
        print("\nSelect configuration for summarization:")
        provider_summary, model_summary = select_provider_and_model("summarization")

        if source_type == "youtube":
            # Process YouTube URL
            while True:
                url = input("\nEnter YouTube URL (or 'q' to quit): ")
                if url.lower() == "q":
                    return
                if url.strip():
                    break
                print("Please enter a valid URL.")

            # Select Whisper model and language option
            model_name, is_english_model = select_whisper_model(is_english_content)

            # Select summary language preference
            use_english_summary = select_summary_language(
                is_english_model, is_english_content
            )

            print(f"Selected model: {model_name}")
            print(f"Is English model: {is_english_model}")
            print(f"Is English content: {is_english_content}")
            print(f"Use English summary: {use_english_summary}")

            # Check and setup whisper.cpp
            if not check_whisper_setup(model_name, is_english_model):
                print("Whisper.cpp setup incomplete. Running setup...")
                setup_whisper(model_name, is_english_model)

            base_filename = generate_output_filename(url)
            audio_path = "temp_audio.wav"

            try:
                # Download and process audio
                print("\nDownloading audio...")
                audio_path = download_audio(url)
                print(
                    f"Audio downloaded: {os.path.getsize(audio_path) / (1024*1024):.1f}MB"
                )

                if os.path.exists(audio_path):
                    size_mb = os.path.getsize(audio_path) / 1024 / 1024
                    print(f"Audio file size: {size_mb:.2f} MB")

                # Transcribe audio with selected model
                print("\nTranscribing audio...")
                transcript = transcribe_audio(audio_path, model_name, is_english_model)

                if not transcript:
                    print("Failed to get transcription")
                    return

                # Clean up the transcript
                cleaned_transcript = clean_transcript(transcript)

                # Generate summary
                print("\nGenerating summary...")
                summary = summarize_text(
                    cleaned_transcript,
                    provider_summary,
                    model_summary,
                    use_english_summary,
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

        else:  # source_type == "transcript"
            # Select and process existing transcript
            transcript_path = select_transcript()
            if not transcript_path:
                print("No transcript selected. Exiting.")
                return

            # Pass the correct is_english_content value from earlier
            use_english_summary = select_summary_language(False, is_english_content)

            base_filename = generate_output_filename_from_path(transcript_path)

            # Generate summary from transcript
            summary = summarize_from_file(
                transcript_path, provider_summary, model_summary, use_english_summary
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

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        if "--debug" in sys.argv:
            raise


if __name__ == "__main__":
    main()
