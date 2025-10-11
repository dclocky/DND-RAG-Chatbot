import os
import torch
import logging
import tiktoken
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "./data")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "google/flan-t5-large")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 250))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 125))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 512))
RETRIEVAL_ALPHA = float(os.getenv("RETRIEVAL_ALPHA", 0.55))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format="%(levelname)s: %(message)s")
logger = logging.getLogger("dnd-rag")

encoding = tiktoken.get_encoding("cl100k_base")
torch.set_default_dtype(torch.float32)

def truncate_to_max_tokens(text: str, max_tokens: int = MAX_TOKENS) -> str:
    tokens = encoding.encode(text)
    return encoding.decode(tokens[:max_tokens]) if len(tokens) > max_tokens else text
