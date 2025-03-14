from groq import Groq
import yt_dlp
from pathlib import Path
import re
import os
from dotenv import load_dotenv
import subprocess
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Pricing data for Whisper models (per hour transcribed, minimum 10s)
WHISPER_PRICING = {
    "whisper-large-v3": 0.111,  # $0.111/hr
    "whisper-large-v3-turbo": 0.04,  # $0.04/hr
    "distil-whisper-large-v3-en": 0.02,  # $0.02/hr
}

# Pricing data for chat model (per million tokens)
CHAT_PRICING = {
    "llama-3.3-70b-versatile": {
        "input": 0.59,
        "output": 0.79,
    }  # $0.59/M input, $0.79/M output
}


# Function to calculate transcription cost
def calculate_transcription_cost(model, duration_seconds):
    price_per_hour = WHISPER_PRICING.get(
        model, 0.02
    )  # Default to Distil-Whisper if unknown
    duration_hours = max(duration_seconds, 10) / 3600  # Minimum 10s charge
    cost = price_per_hour * duration_hours
    return cost


# Function to calculate chat cost
def calculate_chat_cost(model, prompt_tokens, completion_tokens):
    pricing = CHAT_PRICING.get(
        model, {"input": 0.59, "output": 0.79}
    )  # Default to Llama 3.3
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


# Function to stream and process YouTube audio
def stream_youtube_audio(url, output_path="audio/output.mp3"):
    Path(output_path).parent.mkdir(exist_ok=True)
    try:
        command = [
            "yt-dlp",
            url,
            "-o",
            "-",
            "--quiet",
            "--format",
            "bestaudio/best",
            "|",
            "ffmpeg",
            "-i",
            "pipe:0",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-b:a",
            "32k",
            "-f",
            "mp3",
            output_path,
            "-y",
        ]
        subprocess.run(" ".join(command), shell=True, check=True, capture_output=True)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error streaming audio: {e.stderr.decode()}")
        return None


# Function to get audio duration
def get_audio_duration(audio_path):
    result = subprocess.run(
        ["ffmpeg", "-i", audio_path, "-hide_banner"],
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    duration_match = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", result.stderr)
    if duration_match:
        h, m, s = map(float, duration_match.groups())
        return h * 3600 + m * 60 + s
    return None


# Function to split audio into chunks
def split_audio_into_chunks(audio_path, chunk_size_mb=25):
    chunk_dir = Path("audio/chunks")
    chunk_dir.mkdir(exist_ok=True, parents=True)
    base_name = os.path.basename(audio_path).replace(".mp3", "")
    chunk_pattern = str(chunk_dir / f"{base_name}_chunk%03d.mp3")

    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    duration = get_audio_duration(audio_path)
    if not duration:
        print("Could not determine audio duration")
        return None

    chunk_duration = (chunk_size_mb / file_size_mb) * duration
    chunk_duration = max(min(chunk_duration, 600), 60)  # 1-10 min

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                audio_path,
                "-f",
                "segment",
                "-segment_time",
                str(chunk_duration),
                "-c:a",
                "libmp3lame",
                "-b:a",
                "32k",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-y",
                chunk_pattern,
            ],
            check=True,
            capture_output=True,
        )
        return sorted(chunk_dir.glob(f"{base_name}_chunk*.mp3"))
    except subprocess.CalledProcessError as e:
        print(f"Error splitting audio: {e.stderr.decode()}")
        return None


# Function to transcribe audio with verbose_json and cost
def transcribe_audio(audio_path, model="distil-whisper-large-v3-en"):
    try:
        duration = get_audio_duration(audio_path)
        if not duration:
            print("Duration unknown, assuming 10s minimum for cost.")
            duration = 10

        with open(audio_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), audio_file.read()),
                model=model,
                response_format="verbose_json",
            )
            transcription = response.text
            token_count = sum(len(segment["tokens"]) for segment in response.segments)
            cost = calculate_transcription_cost(model, duration)
            print(f"\nToken Usage for {audio_path}:")
            print(f"Estimated tokens: {token_count}")
            print(f"Segment count: {len(response.segments)}")
            print(f"Duration: {duration:.1f}s")
            print(f"Cost: ${cost:.4f}")
            return transcription, token_count, cost
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None, 0, 0


# Function to transcribe chunks in parallel
def transcribe_chunks(chunk_paths, model="distil-whisper-large-v3-en"):
    transcriptions = []
    total_tokens = 0
    total_cost = 0

    def transcribe_single_chunk(chunk_path):
        transcription, token_count, cost = transcribe_audio(chunk_path, model)
        os.remove(chunk_path)
        return transcription, token_count, cost

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
            tqdm(
                executor.map(transcribe_single_chunk, chunk_paths),
                total=len(chunk_paths),
                desc="Transcribing chunks",
            )
        )
        for transcription, token_count, cost in results:
            if transcription:
                transcriptions.append(transcription)
                total_tokens += token_count
                total_cost += cost

    print(f"\nTotal Estimated Token Usage for All Chunks: {total_tokens}")
    print(f"Total Cost for All Chunks: ${total_cost:.4f}")
    return " ".join(transcriptions)


# Function to select a text file
def select_text_file():
    text_dir = Path("text_files")
    text_dir.mkdir(exist_ok=True)
    text_files = list(text_dir.glob("*.txt"))

    if text_files:
        print("\nAvailable text files:")
        for i, file in enumerate(text_files, 1):
            print(f"{i}. {file.name}")
        choice = input(
            f"Enter number 1-{len(text_files)} or 'new' for new file: "
        ).strip()
        if choice.lower() == "new":
            return None
        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(text_files):
                return str(text_files[choice_num - 1])
        except ValueError:
            pass

    file_input = input("Enter text file path or name (e.g., 'myfile.txt'): ").strip()
    file_path = text_dir / file_input if os.sep not in file_input else Path(file_input)
    if not file_path.exists():
        file_path.touch()
        print(f"Created empty file at {file_path}")
    return str(file_path)


# Function to read text file
def read_text_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as text_file:
            return text_file.read()
    except Exception as e:
        print(f"Error reading text file: {e}")
        return None


# Function to get content based on mode
def get_content(mode):
    if mode == "video":
        youtube_url = input("Enter YouTube video URL: ")
        print("Streaming and processing audio...")
        audio_path = stream_youtube_audio(youtube_url)
        if not audio_path:
            return None

        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        print(f"Audio file size: {file_size_mb:.1f}MB")

        if file_size_mb <= 25:
            print("Transcribing audio...")
            content, _, _ = transcribe_audio(audio_path)
            os.remove(audio_path)
        else:
            print("Splitting audio into chunks...")
            chunks = split_audio_into_chunks(audio_path)
            os.remove(audio_path)
            if not chunks:
                return None
            content = transcribe_chunks(chunks)

        if not content:
            print("Failed to transcribe audio.")
            return None

        text_dir = Path("text_files")
        text_dir.mkdir(exist_ok=True)
        filename = f"transcript_{Path(audio_path).stem}.txt"
        file_path = text_dir / filename
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"\nTranscription saved to: {file_path}")
        return content

    elif mode == "text":
        file_path = select_text_file()
        if not file_path:
            return None
        content = read_text_file(file_path)
        if not content:
            print("Text file is empty or failed to read.")
            return "No content available."
        return content
    return None


# Function to chat with persistent context and token tracking
def chat_session(content, mode):
    source_type = "video transcription" if mode == "video" else "text file content"
    print(f"\nChat session started. Ask questions about the {source_type}!")
    print("Type 'exit' to end the session.\n")

    system_prompt = f"""You are an assistant designed to help users explore and understand content from a {source_type}. 
    The content has been provided to you (see below). Your role is to answer questions, provide summaries, extract key points, 
    or assist with any tasks related to this content. If the user asks something unrelated, politely redirect them to the 
    provided {source_type}. Here is the content to work with:\n\n{content}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": "Understood. I'm ready to ask questions about the content.",
        },
    ]
    chat_model = "llama-3.3-70b-versatile"

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Chat session ended.")
            break

        messages.append({"role": "user", "content": user_input})

        try:
            response = client.chat.completions.create(
                model=chat_model, messages=messages, max_tokens=500
            )
            assistant_response = response.choices[0].message.content
            print(f"Assistant: {assistant_response}\n")

            if hasattr(response, "usage"):
                usage = response.usage
                cost = calculate_chat_cost(
                    chat_model, usage.prompt_tokens, usage.completion_tokens
                )
                print("Token Usage for this message:")
                print(f"Prompt tokens: {usage.prompt_tokens}")
                print(f"Completion tokens: {usage.completion_tokens}")
                print(f"Total tokens: {usage.total_tokens}")
                print(f"Total time: {usage.total_time:.3f}s")
                print(f"Cost: ${cost:.6f}\n")

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
