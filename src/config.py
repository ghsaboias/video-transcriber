"""This module contains configuration variables used across the application."""

PROVIDERS_AND_MODELS = {
    "openrouter": {
        "display": "OpenRouter",
        "summary_models": [
            ("Sonnet 3.7", "anthropic/claude-3.7-sonnet"),
            ("Claude 3 Opus", "anthropic/claude-3-opus-20240229"),
            ("Claude 3 Sonnet", "anthropic/claude-3-sonnet-20240229"),
            ("GPT-4 Turbo", "openai/gpt-4-turbo-preview"),
            ("GPT-4o-mini", "openai/gpt-4o-mini"),
            ("Mixtral 8x7B", "mistralai/mixtral-8x7b"),
            ("Llama 2 70B", "meta-llama/llama-2-70b-chat"),
            ("Claude 2", "anthropic/claude-2"),
            ("GPT-3.5 Turbo", "openai/gpt-3.5-turbo"),
            ("Gemini Flash 2.0", "google/gemini-2.0-flash-001"),
            ("Gemini Pro 2.0 Experimental", "google/gemini-2.0-pro-exp-02-05:free"),
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

WHISPER_MODELS = {
    "tiny": {
        "multilingual": {
            "name": "tiny",
            "languages": "100+ languages including Portuguese, Spanish, French, German, Italian, etc.",
        },
        "english_only": "tiny.en",
        "size_mb": 75,
        "description": "Fastest, lowest accuracy",
    },
    "base": {
        "multilingual": {
            "name": "base",
            "languages": "100+ languages including Portuguese, Spanish, French, German, Italian, etc.",
        },
        "english_only": "base.en",
        "size_mb": 142,
        "description": "Fast, better accuracy",
    },
    "small": {
        "multilingual": {
            "name": "small",
            "languages": "100+ languages including Portuguese, Spanish, French, German, Italian, etc.",
        },
        "english_only": "small.en",
        "size_mb": 466,
        "description": "Good balance of speed and accuracy",
    },
    "medium": {
        "multilingual": {
            "name": "medium",
            "languages": "100+ languages including Portuguese, Spanish, French, German, Italian, etc.",
        },
        "english_only": "medium.en",
        "size_mb": 1500,
        "description": "Better accuracy, slower",
    },
    "large-v3": {
        "multilingual": {
            "name": "large-v3",
            "languages": "100+ languages including Portuguese, Spanish, French, German, Italian, etc.",
        },
        "english_only": None,
        "size_mb": 3000,
        "description": "Best accuracy, slowest (recommended for non-English)",
    },
}
