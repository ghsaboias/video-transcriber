import yt_dlp
import subprocess
from pathlib import Path
from typing import Optional
import sys
import re
from anthropic import Anthropic
from datetime import datetime
import json
import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def download_audio(url: str, output_path: str = "temp_audio.wav") -> str:
    """
    Download audio from a YouTube video in WAV format (16 kHz)
    """
    output_path_base = os.path.splitext(output_path)[0]
    temp_wav = f"{output_path_base}_temp.wav"
    final_output = f"{output_path_base}.wav"
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': output_path_base,
    }
    
    # Download the audio first
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    # Rename the downloaded file to temp file
    os.rename(f"{output_path_base}.wav", temp_wav)
    
    # Convert to 16 kHz using FFmpeg
    subprocess.run([
        'ffmpeg', '-y',
        '-i', temp_wav,
        '-ar', '16000',  # Set sample rate to 16 kHz
        '-ac', '1',      # Convert to mono
        '-c:a', 'pcm_s16le',  # Use 16-bit PCM codec
        final_output
    ], check=True)
    
    # Clean up the temporary file
    if os.path.exists(temp_wav):
        os.remove(temp_wav)
    
    return final_output

def transcribe_audio(audio_path: str) -> Optional[str]:
    """
    Transcribe audio file using local whisper.cpp
    """
    original_dir = os.getcwd()  # Store original directory
    try:
        # Convert audio_path to absolute path since we'll change directories
        audio_path_abs = os.path.abspath(audio_path)
        whisper_dir = Path("./whisper.cpp")
        
        if not whisper_dir.exists():
            raise FileNotFoundError(f"whisper.cpp directory not found at {whisper_dir}")
        
        # Change to whisper.cpp directory
        print(f"Changing to directory: {whisper_dir}")
        os.chdir(whisper_dir)
        
        print(f"Current working directory: {os.getcwd()}")
        
        cmd = ["./build/bin/main", "-f", audio_path_abs]
        print(f"Transcribing {audio_path_abs}...")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        return result.stdout
        
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        print(f"Command output: {e.stdout}")
        print(f"Error output: {e.stderr}")
        return None
    except Exception as e:
        print(f"An error occurred during transcription: {str(e)}")
        return None
    finally:
        # Always return to original directory
        os.chdir(original_dir)

def clean_transcript(text: str) -> str:
    """
    Remove timestamps from transcript and clean up the text
    Format: [00:00:00.000 --> 00:00:03.920]   Text
    """
    # Remove timestamp lines and clean up extra whitespace
    cleaned = re.sub(r'\[[0-9:.\s\->\s]+\]\s*', '', text)
    
    # Remove multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # Remove [BLANK_AUDIO] markers
    cleaned = re.sub(r'\[BLANK_AUDIO\]', '', cleaned)
    
    # Remove extra newlines
    cleaned = re.sub(r'\n+', '\n', cleaned)
    
    # Strip leading/trailing whitespace
    cleaned = cleaned.strip()
    
    return cleaned

def summarize_transcript(text: str) -> str:
    """
    Summarize the transcript using Claude
    """
    try:
        client = Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )
        
        prompt = f"""Please provide a concise summary of this transcript.

        <transcript>
        {text}
        </transcript>

        Collect key points and topics of the transcript. Focus on the most important, most relevant, and most interesting points.
        """
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=2000,
            system="You are an expert at summarizing transcripts. You are given a transcript of a video and you are tasked with summarizing the key points of the video. You reply with just the summary, without any introduction. You provide a comprehensive summary of the transcript, including the most important, most relevant, and most interesting points.",
            messages=[
                {
                    "role": "assistant",
                    "content": "1. "
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        review_summary(response.content[0].text)
        return response.content[0].text
        
    except Exception as e:
        print(f"Error during summarization: {str(e)}")
        return "Failed to generate summary."

def review_summary(summary: str) -> str:
    """
    Review the summary and provide feedback
    """
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an expert at proofreading and improving summaries. You are given a summary of a transcript and you are tasked with improving the summary with the goal of increasing     clarity, completeness, and accuracy, correcting names of places and people, and any other relevant aspects. You reply with just the rewritten summary, without any introduction."
            },
            {
                "role": "user",
                "content": "Rewrite the following summary and provide feedback: " + summary,
            },
            {
                "role": "assistant",
                "content": "1. "
            }
        ],
        model="llama-3.3-70b-versatile",
    )

    return chat_completion.choices[0].message.content

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
    
    # Check if executable exists
    if not (whisper_dir / "build/bin/main").exists():
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
        subprocess.run(["git", "clone", "https://github.com/ggerganov/whisper.cpp.git", "./whisper.cpp"], check=True)
    
    # Change to whisper directory
    os.chdir(whisper_dir)
    
    # Download model if not exists
    if not Path("models/ggml-base.en.bin").exists():
        print("Downloading whisper model...")
        subprocess.run(["sh", "./models/download-ggml-model.sh", "base.en"], check=True)
    
    # Build the project
    if not Path("build/bin/main").exists():
        print("Building whisper.cpp...")
        subprocess.run(["cmake", "-B", "build"], check=True)
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
        'quiet': True,
        'no_warnings': True,
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
        title = info.get('fulltitle', info.get('title', 'untitled'))
        # Clean the title to make it filesystem-friendly
        title = re.sub(r'[^\w\s-]', '', title).strip()
        # Replace spaces with underscores and limit length
        title = re.sub(r'\s+', '_', title)[:100]  # Increased length limit to accommodate longer titles
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
            print("\nGenerating summary...")
            summary = summarize_transcript(cleaned_transcript)

            # Review summary
            print("\nReviewing summary...")
            reviewed_summary = review_summary(summary)

            # Re-add '1. ' to the summary
            reviewed_summary = "1. " + reviewed_summary
                
            # Save outputs
            transcript_path, summary_path = save_outputs(cleaned_transcript,reviewed_summary, base_filename)
            
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
