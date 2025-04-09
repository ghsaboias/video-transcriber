from groq import Groq, AsyncGroq
import yt_dlp
from pathlib import Path
import re
import os
from dotenv import load_dotenv
import subprocess
from tqdm import tqdm
import time
import psutil
import sys
import gc
import asyncio
from openai import OpenAI

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
async_client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))
open_router_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

WHISPER_PRICING = {
    "whisper-large-v3": 0.111,
    "whisper-large-v3-turbo": 0.04,
    "distil-whisper-large-v3-en": 0.02,
}

CHAT_PRICING = {
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "google/gemini-2.5-pro-exp-03-25": {
        "input": 0.00,
        "output": 0.00,
    },  # Free during experimental phase
}


def get_performance_metrics():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    cpu_percent = process.cpu_percent(interval=None)
    return memory_mb, cpu_percent


def calculate_transcription_cost(model, duration_seconds):
    price_per_hour = WHISPER_PRICING.get(model, 0.02)
    duration_hours = max(duration_seconds, 10) / 3600
    return price_per_hour * duration_hours


def calculate_chat_cost(model, prompt_tokens, completion_tokens):
    pricing = CHAT_PRICING.get(model, {"input": 0.59, "output": 0.79})
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


def stream_youtube_audio(url, output_path="audio/output.mp3"):
    start_time = time.time()
    Path(output_path).parent.mkdir(exist_ok=True)
    try:
        command = (
            f"yt-dlp {url} -o - --quiet --format bestaudio/best | "
            f"ffmpeg -i pipe:0 -ar 16000 -ac 1 -b:a 16k -af silenceremove=1:0:-40dB -threads 2 -f mp3 {output_path} -y"
        )
        subprocess.run(command, shell=True, check=True, capture_output=True)
        elapsed_time = time.time() - start_time
        memory_mb, cpu_percent = get_performance_metrics()
        print(
            f"Streaming Metrics: Time: {elapsed_time:.2f}s, Memory: {memory_mb:.2f}MB, CPU: {cpu_percent:.1f}%"
        )
        gc.collect()
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error streaming audio: {e.stderr.decode()}")
        return None


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


def split_audio_into_chunks(audio_path, chunk_size_mb=25):
    start_time = time.time()
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
    chunk_duration = max(min(chunk_duration, 600), 60)

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
        chunks = sorted(chunk_dir.glob(f"{base_name}_chunk*.mp3"))
        elapsed_time = time.time() - start_time
        memory_mb, cpu_percent = get_performance_metrics()
        print(
            f"Chunking Metrics: Time: {elapsed_time:.2f}s, Memory: {memory_mb:.2f}MB, CPU: {cpu_percent:.1f}%, Chunks: {len(chunks)}"
        )
        gc.collect()
        return chunks
    except subprocess.CalledProcessError as e:
        print(f"Error splitting audio: {e.stderr.decode()}")
        return None


async def async_transcribe_audio(audio_path, model="distil-whisper-large-v3-en"):
    start_time = time.time()
    try:
        duration = get_audio_duration(audio_path)
        if not duration:
            print("Duration unknown, assuming 10s minimum for cost.")
            duration = 10

        with open(audio_path, "rb") as audio_file:
            response = await async_client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), audio_file.read()),
                model=model,
                response_format="verbose_json",
            )
            transcription = response.text
            token_count = sum(len(segment["tokens"]) for segment in response.segments)
            cost = calculate_transcription_cost(model, duration)
            elapsed_time = time.time() - start_time
            memory_mb, cpu_percent = get_performance_metrics()
            print(f"\nToken Usage for {audio_path}:")
            print(f"Estimated tokens: {token_count}")
            print(f"Segment count: {len(response.segments)}")
            print(f"Duration: {duration:.1f}s")
            print(f"Cost: ${cost:.4f}")
            print(
                f"Transcription Metrics: Time: {elapsed_time:.2f}s, Memory: {memory_mb:.2f}MB, CPU: {cpu_percent:.1f}%"
            )
            gc.collect()
            return transcription, token_count, cost
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None, 0, 0


async def transcribe_chunks(chunk_paths, model="distil-whisper-large-v3-en"):
    start_time = time.time()
    transcriptions = []
    total_tokens = 0
    total_cost = 0

    async def transcribe_single_chunk(chunk_path):
        transcription, token_count, cost = await async_transcribe_audio(
            chunk_path, model
        )
        os.remove(chunk_path)
        return transcription, token_count, cost

    tasks = [transcribe_single_chunk(chunk_path) for chunk_path in chunk_paths]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, tuple) and result[0]:
            transcriptions.append(result[0])
            total_tokens += result[1]
            total_cost += result[2]

    elapsed_time = time.time() - start_time
    memory_mb, cpu_percent = get_performance_metrics()
    print(f"\nTotal Estimated Token Usage for All Chunks: {total_tokens}")
    print(f"Total Cost for All Chunks: ${total_cost:.4f}")
    print(
        f"Chunk Transcription Metrics: Time: {elapsed_time:.2f}s, Memory: {memory_mb:.2f}MB, CPU: {cpu_percent:.1f}%"
    )
    gc.collect()
    return " ".join(transcriptions)


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


def read_text_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as text_file:
            return text_file.read()
    except Exception as e:
        print(f"Error reading text file: {e}")
        return None


async def get_content(mode):
    overall_start_time = time.time()
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
            content, _, _ = await async_transcribe_audio(audio_path)
            os.remove(audio_path)
        else:
            print("Splitting audio into chunks...")
            chunks = split_audio_into_chunks(audio_path)
            os.remove(audio_path)
            if not chunks:
                return None
            content = await transcribe_chunks(chunks)

        if not content:
            print("Failed to transcribe audio.")
            return None

        text_dir = Path("text_files")
        text_dir.mkdir(exist_ok=True)
        filename = f"transcript_{Path(audio_path).stem}.txt"
        file_path = text_dir / filename
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        elapsed_time = time.time() - overall_start_time
        memory_mb, cpu_percent = get_performance_metrics()
        print(f"\nTranscription saved to: {file_path}")
        print(
            f"Overall Video Processing Metrics: Time: {elapsed_time:.2f}s, Memory: {memory_mb:.2f}MB, CPU: {cpu_percent:.1f}%"
        )
        gc.collect()
        return content

    elif mode == "text":
        file_path = select_text_file()
        if not file_path:
            return None
        content = read_text_file(file_path)
        if not content:
            print("Text file is empty or failed to read.")
            return "No content available."
        elapsed_time = time.time() - overall_start_time
        memory_mb, cpu_percent = get_performance_metrics()
        print(
            f"Overall Text Processing Metrics: Time: {elapsed_time:.2f}s, Memory: {memory_mb:.2f}MB, CPU: {cpu_percent:.1f}%"
        )
        gc.collect()
        return content
    return None


def select_chat_model():
    print("\nSelect chat model:")
    print("1. Groq (llama-3.3-70b-versatile)")
    print("2. OpenRouter (google/gemini-2.5-pro-exp-03-25)")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        return "groq", "llama-3.3-70b-versatile"
    elif choice == "2":
        return "openrouter", "google/gemini-2.5-pro-exp-03-25:free"
    else:
        print("Invalid choice. Defaulting to Groq.")
        return "groq", "llama-3.3-70b-versatile"


def chat_session(content, mode):
    source_type = "video transcription" if mode == "video" else "text file content"
    print(f"\nChat session started. Ask questions about the {source_type}!")
    print("Type 'exit' to end the session.\n")

    system_prompt = f"You are here to help with this {source_type}: {content}\nAnswer questions about it using factual information obtained from the video. Do not infer, speculate or guess. If the information is not explicit, be careful with using it. Be careful with roles and relationships mentioned in the text, and only provide information that is directly stated."

    provider, chat_model = select_chat_model()
    total_chat_time = 0
    total_chat_cost = 0
    chat_interactions = 0

    # Initialize messages list for Groq
    messages = [{"role": "system", "content": system_prompt}]

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Chat session ended.")
            if chat_interactions > 0:
                avg_chat_time = total_chat_time / chat_interactions
                print(
                    f"Chat Session Summary: Interactions: {chat_interactions}, Total Cost: ${total_chat_cost:.6f}, Avg Time per Interaction: {avg_chat_time:.2f}s"
                )
            gc.collect()
            break

        messages.append({"role": "user", "content": user_input})

        try:
            start_time = time.time()

            if provider == "groq":
                response = client.chat.completions.create(
                    model=chat_model, messages=messages
                )
            else:  # openrouter
                response = open_router_client.chat.completions.create(
                    model=chat_model,
                    messages=messages,
                    extra_headers={
                        "HTTP-Referer": "http://localhost:3000",
                        "X-Title": "Video Transcriber",
                    },
                )

            assistant_response = response.choices[0].message.content
            elapsed_time = time.time() - start_time
            memory_mb, cpu_percent = get_performance_metrics()

            print(f"Assistant: {assistant_response}\n")
            if hasattr(response, "usage"):
                usage = response.usage
                cost = calculate_chat_cost(
                    chat_model, usage.prompt_tokens, usage.completion_tokens
                )
                total_chat_time += elapsed_time
                total_chat_cost += cost
                chat_interactions += 1
                print("Token Usage for this message:")
                print(f"Prompt tokens: {usage.prompt_tokens}")
                print(f"Completion tokens: {usage.completion_tokens}")
                print(f"Total tokens: {usage.total_tokens}")
                # print(f"Total time: {usage.total_time:.3f}s")
                print(f"Cost: ${cost:.6f}")
                print(
                    f"Chat Metrics: Time: {elapsed_time:.2f}s, Memory: {memory_mb:.2f}MB, CPU: {cpu_percent:.1f}%\n"
                )

            messages.append({"role": "assistant", "content": assistant_response})
            gc.collect()
        except Exception as e:
            print(f"Error in chat response: {e}\n")


async def main():
    print("Choose a mode:")
    print("1. Chat with a YouTube video (transcribes audio)")
    print("2. Chat with a text file")
    mode_choice = input("Enter 1 or 2: ").strip()

    mode = "video" if mode_choice == "1" else "text" if mode_choice == "2" else None
    if not mode:
        print("Invalid choice. Exiting.")
        return

    content = await get_content(mode)
    if content is None:
        print("Failed to obtain content. Exiting.")
        return

    chat_session(content, mode)


if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        print("Please set your GROQ_API_KEY environment variable.")
    elif not os.environ.get("OPENROUTER_API_KEY"):
        print("Please set your OPENROUTER_API_KEY environment variable.")
    else:
        asyncio.run(main())
