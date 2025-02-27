"""This module handles audio transcription using Whisper.cpp."""

import subprocess
import os
from pathlib import Path
from typing import Optional
from src.config import WHISPER_MODELS


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

    # Check if model exists and is valid
    model_info = WHISPER_MODELS[model_name]
    model_type = "english_only" if english_only else "multilingual"
    model_filename = (
        model_info[model_type]["name"]
        if isinstance(model_info[model_type], dict)
        else model_info[model_type]
    )
    model_path = whisper_dir / f"models/ggml-{model_filename}.bin"

    # Check if model exists and has a reasonable size
    if not model_path.exists():
        return False

    # Get expected size from config
    expected_size_mb = model_info["size_mb"]
    actual_size_mb = model_path.stat().st_size / (1024 * 1024)

    # If the model file is significantly smaller than expected (allowing 10% margin),
    # consider it corrupted and trigger a redownload
    if actual_size_mb < expected_size_mb * 0.9:
        print(
            f"Model file appears corrupted (size: {actual_size_mb:.1f}MB, expected: {expected_size_mb}MB)"
        )
        model_path.unlink()  # Delete corrupted model
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

        # Ensure models directory exists
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)

        # Check if download script exists, if not create it
        download_script = models_dir / "download-ggml-model.sh"
        if not download_script.exists():
            print("Creating model download script...")
            download_script_content = """#!/bin/bash

# Whisper model download script

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <model>"
    exit 1
fi

model=$1

# Hugging Face model map
declare -A model_map
model_map["tiny.en"]="ggml-tiny.en.bin"
model_map["tiny"]="ggml-tiny.bin"
model_map["base.en"]="ggml-base.en.bin"
model_map["base"]="ggml-base.bin"
model_map["small.en"]="ggml-small.en.bin"
model_map["small"]="ggml-small.bin"
model_map["medium.en"]="ggml-medium.en.bin"
model_map["medium"]="ggml-medium.bin"
model_map["large-v3"]="ggml-large-v3.bin"

# Check if model exists in map
if [ -z "${model_map[$model]}" ]; then
    echo "Invalid model: $model"
    echo "Available models: ${!model_map[@]}"
    exit 1
fi

# Download model
echo "Downloading ggml model $model from 'https://huggingface.co/ggerganov/whisper.cpp' ..."
wget --quiet --show-progress -O models/ggml-$model.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-$model.bin"""

            # Write the script
            with open(download_script, "w") as f:
                f.write(download_script_content)

            # Make the script executable
            download_script.chmod(0o755)

        # Determine model filename
        model_info = WHISPER_MODELS[model_name]
        model_type = "english_only" if english_only else "multilingual"
        model_filename = (
            model_info[model_type]["name"]
            if isinstance(model_info[model_type], dict)
            else model_info[model_type]
        )

        # Download model if not exists or is corrupted
        model_path = Path(f"models/ggml-{model_filename}.bin")
        if not model_path.exists():
            print(f"Downloading whisper model {model_filename}...")
            subprocess.run(
                ["bash", "./models/download-ggml-model.sh", model_filename], check=True
            )

            # Verify the downloaded model
            if not model_path.exists():
                raise RuntimeError(f"Failed to download model {model_filename}")

            actual_size_mb = model_path.stat().st_size / (1024 * 1024)
            expected_size_mb = model_info["size_mb"]

            if actual_size_mb < expected_size_mb * 0.9:
                raise RuntimeError(
                    f"Downloaded model appears corrupted (size: {actual_size_mb:.1f}MB, expected: {expected_size_mb}MB)"
                )

        # Build the project
        print("Building whisper.cpp...")
        # Create build directory
        Path("build").mkdir(exist_ok=True)

        # Configure CMake
        subprocess.run(
            [
                "cmake",
                "-B",
                "build",
                "-DCMAKE_BUILD_TYPE=Release",
            ],
            check=True,
        )

        # Build the project
        subprocess.run(
            ["cmake", "--build", "build", "--config", "Release", "-j"], check=True
        )

    except Exception as e:
        print(f"Error during whisper.cpp setup: {str(e)}")
        raise
    finally:
        os.chdir(original_dir)
