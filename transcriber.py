"""This module handles audio transcription using Whisper.cpp."""

import subprocess
import os
from pathlib import Path
from typing import Optional
from config import WHISPER_MODELS


def transcribe_audio(
    audio_path: str, model_name: str, english_only: bool = False
) -> Optional[str]:
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

        # Determine model filename
        model_info = WHISPER_MODELS[model_name]
        model_type = "english_only" if english_only else "multilingual"
        model_filename = (
            model_info[model_type]["name"]
            if isinstance(model_info[model_type], dict)
            else model_info[model_type]
        )

        # Use subprocess.Popen to stream output in real-time
        cmd = [
            "./build/bin/whisper-cli",
            "-m",
            f"models/ggml-{model_filename}.bin",
            "-f",
            audio_path_abs,
        ]

        # Add language detection for multilingual mode
        if not english_only:
            cmd.extend(
                ["-l", "auto"]
            )  # Auto detect language and transcribe in original language
            print(
                "Using multilingual model - language will be automatically detected and transcribed in the original language"
            )
        else:
            cmd.extend(["-l", "en"])  # Force English for English-only models
            print("Using English-only model")

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


def check_whisper_setup(model_name: str, english_only: bool = False) -> bool:
    """
    Check if whisper.cpp is properly set up with the specified model
    Returns True if setup is complete, False if setup is needed
    """
    whisper_dir = Path("./whisper.cpp")

    # Check if directory exists
    if not whisper_dir.exists():
        return False

    # Check if model exists
    model_info = WHISPER_MODELS[model_name]
    model_type = "english_only" if english_only else "multilingual"
    model_filename = (
        model_info[model_type]["name"]
        if isinstance(model_info[model_type], dict)
        else model_info[model_type]
    )
    if not (whisper_dir / f"models/ggml-{model_filename}.bin").exists():
        return False

    # Check if executable exists
    if not (whisper_dir / "build/bin/whisper-cli").exists():
        return False

    return True


def setup_whisper(model_name: str, english_only: bool = False):
    """
    Download and setup whisper.cpp with specified model
    """
    original_dir = os.getcwd()
    try:
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

        # Determine model filename
        model_info = WHISPER_MODELS[model_name]
        model_type = "english_only" if english_only else "multilingual"
        model_filename = (
            model_info[model_type]["name"]
            if isinstance(model_info[model_type], dict)
            else model_info[model_type]
        )

        # Download model if not exists
        model_path = Path(f"models/ggml-{model_filename}.bin")
        if not model_path.exists():
            print(f"Downloading whisper model {model_filename}...")
            subprocess.run(
                ["sh", "./models/download-ggml-model.sh", model_filename], check=True
            )

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
                    "-DBUILD_SHARED_LIBS=OFF",
                    "-DCMAKE_BUILD_TYPE=Release",
                ],
                check=True,
            )

            # Build the project
            subprocess.run(
                ["cmake", "--build", "build", "--config", "Release"], check=True
            )

    finally:
        os.chdir(original_dir)
