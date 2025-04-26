from dotenv import load_dotenv
load_dotenv()
import os

# MODEL CONFIGURATION
MODEL_PROVIDER = "local"  # Options: "local", "openai"

# For local model
LOCAL_MODEL_NAME = "all-MiniLM-L6-v2"

# For OpenAI
OPENAI_MODEL_NAME = "gpt-3.5-turbo"  # or "gpt-4"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # set this via .env or export
