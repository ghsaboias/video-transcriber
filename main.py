from groq import Groq
import yt_dlp
from pathlib import Path
import re
import os
from dotenv import load_dotenv
import subprocess

load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


# Function to download YouTube audio
def download_youtube_audio(url, output_path="audio"):
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "outtmpl": f"{output_path}/%(title)s.%(ext)s",
        "quiet": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            mp3_filename = re.sub(r"\.[^.]+$", ".mp3", filename)
            return mp3_filename
    except Exception as e:
        print(f"Error downloading audio: {e}")
        return None


# Function to transcribe audio
def transcribe_audio(audio_path):
    try:
        with open(audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model="distil-whisper-large-v3-en",
                response_format="text",
            )
        return transcription
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None


# Function to read text file
def read_text_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as text_file:
            content = text_file.read()
        return content
    except Exception as e:
        print(f"Error reading text file: {e}")
        return None


# Function to select a text file
def select_text_file():
    text_dir = Path("text_files")
    text_dir.mkdir(exist_ok=True)
    
    # List available text files
    text_files = list(text_dir.glob("*.txt"))
    
    if text_files:
        print("\nAvailable text files:")
        for i, file in enumerate(text_files, 1):
            print(f"{i}. {file.name}")
        print(f"{len(text_files) + 1}. Use a different file")
        
        choice = input(f"Enter number 1-{len(text_files) + 1}: ").strip()
        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(text_files):
                return str(text_files[choice_num - 1])
            elif choice_num == len(text_files) + 1:
                # User wants to specify a different file
                pass
            else:
                print("Invalid choice. Please specify a file manually.")
        except ValueError:
            print("Invalid input. Please specify a file manually.")
    else:
        print("No text files found in the 'text_files' directory.")
    
    # If we get here, either there were no files or the user chose to specify a different file
    print(
        "Enter the full path to your .txt file or a filename to create in 'text_files' directory."
    )
    print(
        "Example: '/Users/user/Documents/file.txt' or just 'myfile.txt' to use/create in 'text_files'"
    )
    file_input = input("Text file: ").strip()
    if os.path.splitext(file_input)[1].lower() == ".txt" and os.sep not in file_input:
        file_path = text_dir / file_input
    else:
        file_path = Path(file_input)
    if not file_path.exists():
        print(
            f"File '{file_path}' does not exist. Would you like to create an empty one? (yes/no)"
        )
        if input().lower() == "yes":
            file_path.touch()
            print(
                f"Created empty file at {file_path}. You can add content to it later."
            )
        else:
            print("No file selected.")
            return None
    elif file_path.suffix.lower() != ".txt":
        print("Please select a .txt file.")
        return None
    return str(file_path)


# Function to preprocess audio
def preprocess_audio(input_path):
    """Preprocess audio file to meet Groq API requirements using ffmpeg."""
    output_path = input_path.replace(".mp3", "_processed.mp3")
    try:
        # Convert to 16kHz mono MP3 with low bitrate
        subprocess.run([
            "ffmpeg", "-i", input_path,
            "-ar", "16000",    # Sample rate: 16kHz (optimal for speech)
            "-ac", "1",        # Mono audio
            "-b:a", "32k",     # Very low bitrate (32kbps)
            "-map", "0:a",     # Audio stream only
            "-y",              # Overwrite output file if exists
            output_path
        ], check=True, capture_output=True)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error preprocessing audio: {e.stderr.decode()}")
        return None


# Function to split audio into chunks
def split_audio_into_chunks(audio_path, chunk_size_mb=20):
    """Split audio into chunks smaller than chunk_size_mb."""
    chunk_dir = Path("audio/chunks")
    chunk_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate unique base name for chunks
    base_name = os.path.basename(audio_path).replace(".mp3", "")
    chunk_base = str(chunk_dir / base_name)
    
    # Use ffmpeg to get duration
    result = subprocess.run([
        "ffmpeg", "-i", audio_path, 
        "-hide_banner"
    ], stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    
    duration_match = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", result.stderr)
    if not duration_match:
        print("Could not determine audio duration")
        return None
    
    # Convert duration to seconds
    h, m, s = map(float, duration_match.groups())
    total_duration = h * 3600 + m * 60 + s
    
    # Estimate bytes per second (approximate)
    file_size = os.path.getsize(audio_path)
    bytes_per_sec = file_size / total_duration
    
    # Calculate chunk duration to stay under chunk_size_mb
    chunk_bytes = chunk_size_mb * 1024 * 1024  # Convert MB to bytes
    chunk_duration = chunk_bytes / bytes_per_sec
    
    # Ensure chunk size is reasonable (between 1-10 minutes)
    chunk_duration = max(min(chunk_duration, 600), 60)
    
    chunks = []
    # Split file into chunks
    for i in range(0, int(total_duration), int(chunk_duration)):
        chunk_path = f"{chunk_base}_chunk{i}.mp3"
        
        # Execute ffmpeg to extract chunk
        try:
            subprocess.run([
                "ffmpeg", "-i", audio_path,
                "-ss", str(i),
                "-t", str(chunk_duration),
                "-c:a", "libmp3lame",
                "-b:a", "32k",
                "-ar", "16000",
                "-ac", "1",
                "-y", chunk_path
            ], check=True, capture_output=True)
            
            chunks.append(chunk_path)
        except subprocess.CalledProcessError as e:
            print(f"Error creating chunk: {e.stderr.decode()}")
    
    return chunks


# Function to transcribe multiple chunks
def transcribe_chunks(chunk_paths):
    """Transcribe multiple audio chunks and combine them."""
    transcriptions = []
    
    for i, chunk_path in enumerate(chunk_paths):
        print(f"Transcribing chunk {i+1}/{len(chunk_paths)}...")
        transcript = transcribe_audio(chunk_path)
        if transcript:
            transcriptions.append(transcript)
        # Clean up chunk
        os.remove(chunk_path)
    
    # Combine all transcriptions
    return " ".join(transcriptions)


# Function to get content based on mode
def get_content(mode):
    if mode == "video":
        youtube_url = input("Enter YouTube video URL: ")
        Path("audio").mkdir(exist_ok=True)

        # Get video title before downloading
        print("Fetching video information...")
        ydl_opts = {"quiet": True, "no_warnings": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            video_title = info.get("title", "")

        print("Downloading audio...")
        audio_path = download_youtube_audio(youtube_url)
        if not audio_path:
            print("Failed to download audio.")
            return None

        original_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        print(f"Original audio file size: {original_size_mb:.1f}MB")
        
        # If file is already small enough, use it directly
        if original_size_mb <= 25:
            print("Audio file size is within limits, proceeding with transcription...")
        else:
            # Try preprocessing first
            print("Preprocessing audio to reduce file size...")
            processed_path = preprocess_audio(audio_path)
            if not processed_path:
                print("Failed to preprocess audio.")
                os.remove(audio_path)
                return None
            
            processed_size_mb = os.path.getsize(processed_path) / (1024 * 1024)
            print(f"Processed audio file size: {processed_size_mb:.1f}MB")
            
            # If preprocessing reduced size enough, use processed file
            if processed_size_mb <= 25:
                print("Preprocessing successful, proceeding with transcription...")
                os.remove(audio_path)
                audio_path = processed_path
            else:
                # If file is still too large, try splitting into chunks
                print("Audio file still too large after preprocessing.")
                print("Splitting audio into smaller chunks...")
                chunks = split_audio_into_chunks(audio_path)
                
                # Clean up
                os.remove(audio_path)
                if processed_path != audio_path:
                    os.remove(processed_path)
                
                if not chunks or len(chunks) == 0:
                    print("Failed to split audio into chunks.")
                    return None
                
                print(f"Successfully split audio into {len(chunks)} chunks")
                content = transcribe_chunks(chunks)
                
                if not content:
                    print("Failed to transcribe audio chunks.")
                    return None
                
                print("\nTranscription completed:")
                print(f"{content[:500]}..." if len(content) > 500 else content)
                
                # Save transcription (using same code as below)
                text_dir = Path("text_files")
                text_dir.mkdir(exist_ok=True)
                
                # Use first 5 words of video title for filename
                title_words = video_title.split()[:5]
                title_slug = "_".join(title_words)
                title_slug = re.sub(r"[^\w\s-]", "", title_slug).strip().lower()
                title_slug = re.sub(r"[-\s]+", "_", title_slug)
                
                filename = f"{title_slug}_transcript.txt"
                
                file_path = text_dir / filename
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                
                print(f"\nTranscription saved to: {file_path}")
                return content

        # Normal transcription for files under limit
        print("Transcribing audio...")
        content = transcribe_audio(audio_path)
        if not content:
            print("Failed to transcribe audio.")
            os.remove(audio_path)
            return None
        
        print("\nTranscription completed:")
        print(f"{content[:500]}..." if len(content) > 500 else content)

        # Save transcription to text_files directory
        text_dir = Path("text_files")
        text_dir.mkdir(exist_ok=True)

        # Use first 5 words of video title for filename
        title_words = video_title.split()[:5]
        title_slug = "_".join(title_words)
        # Remove special characters that might cause issues in filenames
        title_slug = re.sub(r"[^\w\s-]", "", title_slug).strip().lower()
        title_slug = re.sub(r"[-\s]+", "_", title_slug)

        filename = f"{title_slug}_transcript.txt"

        file_path = text_dir / filename
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"\nTranscription saved to: {file_path}")

        print("\nCleaning up downloaded audio file...")
        os.remove(audio_path)
        return content
    elif mode == "text":
        file_path = select_text_file()
        if not file_path:
            return None
        print("Reading text file...")
        content = read_text_file(file_path)
        if not content:
            print("Text file is empty or failed to read. Proceeding with no content.")
            content = "No content available."
        print("\nText content loaded (showing first 500 characters):")
        print(f"{content[:500]}..." if len(content) > 500 else content)
        return content
    return None


# Function to chat with persistent context
def chat_session(content, mode):
    source_type = "video transcription" if mode == "video" else "text file content"
    print(f"\nChat session started. Ask questions about the {source_type}!")
    print("Type 'exit' to end the session.\n")

    # Initialize the messages array
    messages = [
        {
            "role": "system",
            "content": f"You are a helpful assistant that answers questions based on {source_type}.",
        },
        {
            "role": "user",
            "content": f"Here is the {source_type} to base your answers on:\n\n{content}",
        },
    ]

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chat session ended.")
            break

        # Append user input to messages
        messages.append({"role": "user", "content": user_input})

        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile", messages=messages, max_tokens=500
            )
            assistant_response = response.choices[0].message.content
            print(f"Assistant: {assistant_response}\n")

            # Append assistant response to messages
            messages.append({"role": "assistant", "content": assistant_response})
        except Exception as e:
            print(f"Error in chat response: {e}\n")


# Main function
def main():
    print("Choose a mode:")
    print("1. Chat with a YouTube video (transcribes audio)")
    print("2. Chat with a text file")
    mode_choice = input("Enter 1 or 2: ").strip()

    mode = "video" if mode_choice == "1" else "text" if mode_choice == "2" else None
    if not mode:
        print("Invalid choice. Exiting.")
        return

    content = get_content(mode)
    if content is None:
        print("Failed to obtain content. Exiting.")
        return

    chat_session(content, mode)


if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        print("Please set your GROQ_API_KEY environment variable.")
    else:
        main()
