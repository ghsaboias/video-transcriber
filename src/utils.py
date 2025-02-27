"""This module contains utility functions for file management and other helper tasks."""

from pathlib import Path
import os
import yt_dlp
import re


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
        title = re.sub(r"\s+", "_", title)[:100]
    else:
        print("Could not get video title, using 'untitled'")
        title = "untitled"

    return title


def generate_output_filename_from_path(file_path: str) -> str:
    """
    Generate a filename from the input transcript file path
    """
    # Get the base name without extension
    base_name = Path(file_path).stem
    # Clean the name to make it filesystem-friendly
    clean_name = re.sub(r"[^\w\s-]", "", base_name).strip()
    # Replace spaces with underscores and limit length
    clean_name = re.sub(r"\s+", "_", clean_name)[:100]
    return clean_name


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


def list_available_transcripts() -> list:
    """
    List all available transcripts in the outputs/transcripts directory
    Returns a list of transcript files
    """
    transcript_dir = Path("outputs/transcripts")
    if not transcript_dir.exists():
        return []

    transcripts = list(transcript_dir.glob("*.txt"))
    return transcripts
