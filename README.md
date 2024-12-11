# Video Transcriber

A Python tool that locally transcribes YouTube videos using whisper.cpp and generates summaries using Claude AI.

## Features

- Downloads audio from YouTube videos
- Transcribes audio using local whisper.cpp
- Generates summaries using Claude AI
- Organizes transcripts and summaries in a clean directory structure
- Handles video titles for easy file organization

## Prerequisites

- Python 3.6+
- git
- cmake
- C++ compiler (gcc/clang/MSVC)
- FFmpeg

### System Dependencies

Ubuntu/Debian:
```
sudo apt-get install git cmake build-essential ffmpeg
```

macOS with Homebrew:
```
brew install git cmake ffmpeg
```

## Installation

1. Clone the repository:
```
git clone https://github.com/ghsaboias/video-transcriber.git
cd video-transcriber
```

2. Create and activate a virtual environment:
```
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install Python dependencies:
```
pip install yt-dlp anthropic
```

4. Set up your environment variables in a `.env` file:
```
ANTHROPIC_API_KEY=your_claude_api_key_here
```

## Usage

1. Run the script:
```
python main.py
```

2. Enter a YouTube URL when prompted

3. The script will:
   - Download the audio
   - Transcribe it using whisper.cpp
   - Generate a summary using Claude
   - Save both files in organized directories:
     - `outputs/transcripts/[video_title].txt`
     - `outputs/summaries/[video_title].txt`

## Output Structure

```
outputs/
  ├── transcripts/
  │   └── [video_title].txt
  └── summaries/
      └── [video_title].txt
```

## Dependencies

- [yt-dlp](https://github.com/yt-dlp/yt-dlp): For downloading YouTube videos
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp): For transcription
- [Anthropic Claude](https://www.anthropic.com/): For generating summaries
- [FFmpeg](https://ffmpeg.org/): For audio processing

## License

MIT

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper)
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
- [Anthropic Claude](https://www.anthropic.com/)
