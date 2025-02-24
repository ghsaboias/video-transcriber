import yt_dlp
import subprocess
from pathlib import Path
from typing import Optional, Tuple
import sys
import re
import os
from openai import OpenAI
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Dictionary to store available providers and their models.
PROVIDERS_AND_MODELS = {
    "openrouter": {
        "display": "OpenRouter",
        "summary_models": [
            ("Claude 3 Opus", "anthropic/claude-3-opus-20240229"),
            ("Claude 3 Sonnet", "anthropic/claude-3-sonnet-20240229"),
            ("GPT-4 Turbo", "openai/gpt-4-turbo-preview"),
            ("Mixtral 8x7B", "mistralai/mixtral-8x7b"),
            ("Llama 2 70B", "meta-llama/llama-2-70b-chat"),
            ("Claude 2", "anthropic/claude-2"),
            ("GPT-3.5 Turbo", "openai/gpt-3.5-turbo"),
            ("Gemini Flash 2.0", "google/gemini-2.0-flash-001"),
        ],
    },
    "groq": {
        "display": "Groq",
        "summary_models": [
            ("distil-whisper-large-v3-en", "distil-whisper-large-v3-en"),
            ("gemma2-9b-it", "gemma2-9b-it"),
            ("llama-3.3-70b-versatile", "llama-3.3-70b-versatile"),
            ("llama-3.1-8b-instant", "llama-3.1-8b-instant"),
            ("llama-guard-3-8b", "llama-guard-3-8b"),
            ("llama3-70b-8192", "llama3-70b-8192"),
            ("llama3-8b-8192", "llama3-8b-8192"),
            ("mixtral-8x7b-32768", "mixtral-8x7b-32768"),
            ("whisper-large-v3", "whisper-large-v3"),
            ("whisper-large-v3-turbo", "whisper-large-v3-turbo"),
            # Preview Models
            (
                "deepseek-r1-distill-llama-70b-specdec",
                "deepseek-r1-distill-llama-70b-specdec",
            ),
            ("deepseek-r1-distill-llama-70b", "deepseek-r1-distill-llama-70b"),
            ("llama-3.3-70b-specdec", "llama-3.3-70b-specdec"),
            ("llama-3.2-1b-preview", "llama-3.2-1b-preview"),
            ("llama-3.2-3b-preview", "llama-3.2-3b-preview"),
            ("llama-3.2-11b-vision-preview", "llama-3.2-11b-vision-preview"),
            ("llama-3.2-90b-vision-preview", "llama-3.2-90b-vision-preview"),
        ],
    },
}


def download_audio(url: str, output_path: str = "temp_audio.wav") -> str:
    """
    Download audio from a YouTube video in WAV format (16 kHz)
    """
    output_path_base = os.path.splitext(output_path)[0]
    temp_wav = f"{output_path_base}_temp.wav"
    final_output = f"{output_path_base}.wav"

    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "outtmpl": output_path_base,
    }

    # Download the audio first
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Rename the downloaded file to temp file
    os.rename(f"{output_path_base}.wav", temp_wav)

    # Convert to 16 kHz using FFmpeg
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            temp_wav,
            "-ar",
            "16000",  # Set sample rate to 16 kHz
            "-ac",
            "1",  # Convert to mono
            "-c:a",
            "pcm_s16le",  # Use 16-bit PCM codec
            final_output,
        ],
        check=True,
    )

    # Clean up the temporary file
    if os.path.exists(temp_wav):
        os.remove(temp_wav)

    return final_output


def transcribe_audio(audio_path: str) -> Optional[str]:
    """
    Transcribe audio file using local whisper.cpp with Metal acceleration
    """
    original_dir = os.getcwd()  # Store original directory
    try:
        audio_path_abs = os.path.abspath(audio_path)
        whisper_dir = Path("./whisper.cpp")

        if not whisper_dir.exists():
            raise FileNotFoundError(f"whisper.cpp directory not found at {whisper_dir}")

        os.chdir(whisper_dir)
        print(f"Changed to whisper.cpp directory: {os.getcwd()}")

        # Use subprocess.Popen to stream output in real-time
        cmd = ["./build/bin/whisper-cli", "-f", audio_path_abs]
        print(f"Starting transcription of {audio_path_abs} with Metal acceleration...")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line-buffered
        )

        transcription = []
        # Stream output as it comes
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(f"Transcription progress: {line.strip()}")  # Show live output
                transcription.append(line)

        # Check for errors
        stderr_output = process.stderr.read()
        if process.returncode != 0:
            print(f"Transcription failed with exit code {process.returncode}")
            print(f"Error output: {stderr_output}")
            return None

        print("Transcription completed.")
        return "".join(transcription)

    except Exception as e:
        print(f"An error occurred during transcription: {str(e)}")
        return None
    finally:
        os.chdir(original_dir)


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


def check_whisper_setup() -> bool:
    """
    Check if whisper.cpp is properly set up
    Returns True if setup is complete, False if setup is needed
    """
    whisper_dir = Path("./whisper.cpp")

    # Check if directory exists
    if not whisper_dir.exists():
        return False

    # Check if model exists
    if not (whisper_dir / "models/ggml-base.en.bin").exists():
        return False

    # Check if executable exists (updated path)
    if not (whisper_dir / "build/bin/whisper-cli").exists():
        return False

    return True


def setup_whisper():
    """
    Download and setup whisper.cpp if not already present
    """
    whisper_dir = Path("./whisper.cpp")

    # Clone repository if not exists
    if not whisper_dir.exists():
        print("Cloning whisper.cpp repository...")
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/ggerganov/whisper.cpp.git",
                "./whisper.cpp",
            ],
            check=True,
        )

    # Change to whisper directory
    os.chdir(whisper_dir)

    # Download model if not exists
    if not Path("models/ggml-base.en.bin").exists():
        print("Downloading whisper model...")
        subprocess.run(["sh", "./models/download-ggml-model.sh", "base.en"], check=True)

    # Build the project with static linking
    if not Path("build/bin/main").exists():
        print("Building whisper.cpp...")
        # Create build directory
        Path("build").mkdir(exist_ok=True)

        # Configure CMake with static library
        subprocess.run(
            [
                "cmake",
                "-B",
                "build",
                "-DBUILD_SHARED_LIBS=OFF",  # Build static library instead of shared
                "-DCMAKE_BUILD_TYPE=Release",
            ],
            check=True,
        )

        # Build the project
        subprocess.run(["cmake", "--build", "build", "--config", "Release"], check=True)

    # Change back to original directory
    os.chdir("..")


def create_output_directories():
    """
    Create directories for storing transcripts and summaries
    """
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/transcripts").mkdir(exist_ok=True)
    Path("outputs/summaries").mkdir(exist_ok=True)


def get_video_info(url: str) -> dict:
    """
    Get video information from YouTube URL and print it for debugging
    """
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            return info
        except Exception as e:
            print(f"Warning: Could not get video info ({str(e)})")
            return {}


def generate_output_filename(url: str) -> str:
    """
    Generate a filename from the YouTube video title
    """
    # Get video info and print it
    info = get_video_info(url)

    if info:
        # Use fulltitle if available, fallback to title, then to untitled
        title = info.get("fulltitle", info.get("title", "untitled"))
        # Clean the title to make it filesystem-friendly
        title = re.sub(r"[^\w\s-]", "", title).strip()
        # Replace spaces with underscores and limit length
        title = re.sub(r"\s+", "_", title)[
            :100
        ]  # Increased length limit to accommodate longer titles
    else:
        print("Could not get video title, using 'untitled'")
        title = "untitled"

    return title


def save_outputs(transcript: str, reviewed_summary: str, base_filename: str):
    """
    Save transcript and summary to their respective directories
    """
    # Save transcript
    transcript_path = f"outputs/transcripts/{base_filename}.txt"
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript)

    # Save summary
    summary_path = f"outputs/summaries/{base_filename}.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(reviewed_summary)

    return transcript_path, summary_path


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


def main():
    try:
        # Create output directories
        create_output_directories()

        # Check and setup whisper.cpp if needed
        if not check_whisper_setup():
            print("Whisper.cpp setup incomplete. Running setup...")
            setup_whisper()

        # Get YouTube URL from user
        url = input("Enter YouTube URL: ")

        # Prompt for provider/model selections for summarization.
        print("\nSelect configuration for summarization:")
        provider_summary, model_summary = select_provider_and_model("summarization")

        # Generate base filename for outputs
        base_filename = generate_output_filename(url)

        # Initialize audio_path
        audio_path = "temp_audio.wav"

        try:
            # Download audio
            print("Downloading audio...")
            audio_path = download_audio(url)
            print(f"Audio downloaded to: {audio_path}")

            # Verify file exists and size
            if os.path.exists(audio_path):
                size_mb = os.path.getsize(audio_path) / 1024 / 1024
                print(f"Audio file size: {size_mb:.2f} MB")

            # Transcribe audio
            transcript = transcribe_audio(audio_path)

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

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
