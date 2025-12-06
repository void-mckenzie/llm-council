"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv

load_dotenv()

# OpenRouter API key
OPENROUTER_API_KEY = "sk-or-v1-..."

# Local Model Configuration
LOCAL_MODEL_BASE_URL = os.getenv("LOCAL_MODEL_BASE_URL", "http://localhost:10105/v1/chat/completions")
LOCAL_MODEL_API_KEY = os.getenv("LOCAL_MODEL_API_KEY", "lm-studio")

# Execution Strategy
ROUND_ROBIN_EXECUTION = os.getenv("ROUND_ROBIN_EXECUTION", "True").lower() == "true"

# Council members - list of OpenRouter model identifiers
COUNCIL_MODELS = [
    "local/qwen3-14b-128k-ud-q5_k_xl",
    "local/qwen3-30b-a3b-thinking-2507-ud-q6_k_xl",
    "local/qwen3-32b-128k-ud-q4_k_xl"
]

# Title generation model
TITLE_MODEL = "google/gemini-2.5-flash"

# Chairman model - synthesizes final response
CHAIRMAN_MODEL = "local/qwen3-32b-128k-ud-q4_k_xl"

# OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Data directory for conversation storage
DATA_DIR = "data/conversations"
