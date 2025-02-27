"""This module contains functions for downloading audio from YouTube."""

import yt_dlp
import subprocess
import os


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
