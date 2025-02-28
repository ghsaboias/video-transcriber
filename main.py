"""This is the main entry point of the application."""

import sys
import os
from dotenv import load_dotenv
from src.utils import (
    create_output_directories,
    generate_output_filename,
    save_outputs,
    generate_output_filename_from_path,
    get_video_info,
)
from src.user_interface import (
    select_content_source,
    select_provider_and_model,
    select_whisper_model,
    select_transcript,
    select_overview_language,
)
from src.audio_downloader import download_audio
from src.transcriber import transcribe_audio, check_whisper_setup, setup_whisper
from src.text_processor import (
    clean_transcript,
    generate_detailed_overview,
    generate_overview_from_file,
)

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
        metadata_string = ""  # Initialize metadata_string with a default empty value

        if source_type == "youtube":
            # Process YouTube URL
            while True:
                url = input("\nEnter YouTube URL (or 'q' to quit): ")
                if url.lower() == "q":
                    return
                if url.strip():
                    break
                print("Please enter a valid URL.")

            # Get video metadata for the detailed overview
            video_info = get_video_info(url)
            channel_name = "Unknown Channel"
            video_title = "Untitled Video"

            if video_info:
                # Extract channel name with fallbacks, defaulting to "Unknown Channel" if all are None
                temp_channel = video_info.get(
                    "channel", video_info.get("uploader", None)
                )
                if temp_channel:
                    channel_name = temp_channel

                # Extract video title, defaulting to "Untitled Video" if None
                temp_title = video_info.get("title", None)
                if temp_title:
                    video_title = temp_title

                print(f"Video metadata retrieved: {channel_name} - {video_title}")
            else:
                print("Could not retrieve video metadata, using default values")

            metadata_string = f"Channel: {channel_name}\nVideo Title: {video_title}\n\n"

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

            # Extract metadata from transcript file
            # Assuming first line is channel name and second line is video title
            lines = transcript.split("\n")
            if len(lines) >= 2:
                # Check if the first line starts with "Channel:" and second with "Video Title:"
                if lines[0].startswith("Channel:") and lines[1].startswith(
                    "Video Title:"
                ):
                    channel_line = lines[0]
                    title_line = lines[1]
                    metadata_string = f"{channel_line}\n{title_line}\n\n"
                    print(
                        f"Extracted metadata from transcript: {channel_line} - {title_line}"
                    )
                else:
                    print(
                        "Transcript doesn't contain expected metadata format in first two lines"
                    )
                    metadata_string = ""
            else:
                print("Transcript doesn't have enough lines to contain metadata")
                metadata_string = ""

        if transcript and transcript_path:
            # Only ask about detailed overview generation for YouTube videos
            overview_choice = "1"  # Default to yes for transcript source
            if source_type == "youtube":
                while True:
                    overview_choice = input(
                        "\nWould you like to generate a detailed overview?\n1. Yes\n2. No\nEnter your choice: "
                    ).lower()
                    if overview_choice in ["1", "2"]:
                        break
                    print("Please enter '1' for yes or '2' for no.")

            if overview_choice == "2":
                # Add metadata at the beginning of the script
                transcript = metadata_string + transcript
                with open(transcript_path, "w", encoding="utf-8") as f:
                    f.write(transcript)
                print(
                    "\nSkipping detailed overview generation. Your transcript is ready!"
                )
                return

            # Now that we have the transcript and user wants a detailed overview, configure generation
            print("\nSelect configuration for detailed overview generation:")
            provider_overview, model_overview = select_provider_and_model(
                "detailed_overview"
            )

            # Select overview language preference
            use_english_overview = select_overview_language(
                is_english_model if source_type == "youtube" else False,
                is_english_content,
            )

            # Generate detailed overview
            print("\nGenerating detailed overview...")
            overview = (
                generate_detailed_overview(
                    transcript, provider_overview, model_overview, use_english_overview
                )
                if source_type == "youtube"
                else generate_overview_from_file(
                    transcript_path,
                    provider_overview,
                    model_overview,
                    use_english_overview,
                )
            )

            # Add channel name and video title to the transcript
            # Only add metadata if it exists (for YouTube source)
            if metadata_string and source_type == "youtube":
                transcript = metadata_string + transcript
                with open(transcript_path, "w", encoding="utf-8") as f:
                    f.write(transcript)
                print(
                    f"Transcript updated with channel name and video title: {transcript_path}"
                )

            if overview:
                # Add channel name and video title to the overview
                # Only add metadata if it exists
                if metadata_string:
                    overview = metadata_string + overview

                # Save detailed overview
                overview_path = os.path.join(
                    "outputs", "overviews", f"{base_filename}_overview.txt"
                )
                os.makedirs(os.path.dirname(overview_path), exist_ok=True)
                with open(overview_path, "w", encoding="utf-8") as f:
                    f.write(overview)
                print(f"Detailed overview saved to {overview_path}")
            else:
                print("Failed to generate detailed overview")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        if "--debug" in sys.argv:
            raise


if __name__ == "__main__":
    main()
