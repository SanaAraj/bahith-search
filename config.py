import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Omartificial-Intelligence-Space/Matryoshka-Arabic-STS-V1")
FALLBACK_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SEARCH_ALPHA = float(os.getenv("SEARCH_ALPHA", "0.7"))
TOP_K = int(os.getenv("TOP_K", "5"))
CHROMA_PATH = "chroma_db"
DATA_PATH = "data"
