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

        transcript = None
        base_filename = None
        transcript_path = None

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

            print(f"Selected model: {model_name}")
            print(f"Is English model: {is_english_model}")
            print(f"Is English content: {is_english_content}")

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
                transcript = clean_transcript(transcript)

                # Save transcript immediately
                transcript_path = os.path.join(
                    "outputs", "transcripts", f"{base_filename}.txt"
                )
                os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
                with open(transcript_path, "w", encoding="utf-8") as f:
                    f.write(transcript)
                print(f"\nTranscription saved to {transcript_path}")

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

            base_filename = generate_output_filename_from_path(transcript_path)

            # Read the transcript
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript = f.read()

        if transcript and transcript_path:
            # Only ask about summary generation for YouTube videos
            summarize_choice = "1"  # Default to yes for transcript source
            if source_type == "youtube":
                while True:
                    summarize_choice = input(
                        "\nWould you like to generate a summary?\n1. Yes\n2. No\nEnter your choice: "
                    ).lower()
                    if summarize_choice in ["1", "2"]:
                        break
                    print("Please enter '1' for yes or '2' for no.")

            if summarize_choice == "2":
                print("\nSkipping summary generation. Your transcript is ready!")
                return

            # Now that we have the transcript and user wants a summary, configure summarization
            print("\nSelect configuration for summarization:")
            provider_summary, model_summary = select_provider_and_model("summarization")

            # Select summary language preference
            use_english_summary = select_summary_language(
                is_english_model if source_type == "youtube" else False,
                is_english_content,
            )

            # Generate summary
            print("\nGenerating summary...")
            summary = (
                summarize_text(
                    transcript,
                    provider_summary,
                    model_summary,
                    use_english_summary,
                )
                if source_type == "youtube"
                else summarize_from_file(
                    transcript_path,
                    provider_summary,
                    model_summary,
                    use_english_summary,
                )
            )

            if summary:
                # Save summary
                summary_path = os.path.join(
                    "outputs", "summaries", f"{base_filename}_summary.txt"
                )
                os.makedirs(os.path.dirname(summary_path), exist_ok=True)
                with open(summary_path, "w", encoding="utf-8") as f:
                    f.write(summary)
                print(f"Summary saved to {summary_path}")
            else:
                print("Failed to generate summary")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        if "--debug" in sys.argv:
            raise


if __name__ == "__main__":
    main()
