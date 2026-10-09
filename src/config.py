from pathlib import Path
import os
import torch

HF_CACHE_ROOT = os.getenv("HF_HOME", "/app/.cache/huggingface")

# Persistent cache locations — used by the Dockerfile and docker-compose.
HF_HUB_CACHE = os.path.join(HF_CACHE_ROOT, "hub")
HF_TRANSFORMERS_CACHE = os.path.join(HF_CACHE_ROOT, "transformers")
HF_SENTENCE_TRANSFORMERS_HOME = os.path.join(HF_CACHE_ROOT, "sentence_transformers")

# Device selection — falls back to CPU when CUDA is unavailable.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC_DIR = PROJECT_ROOT / "src"

DATA_DIR = SRC_DIR / "italian_universities"

INDEX_CACHE_DIR = PROJECT_ROOT / "index_cache"
INDEX_PIPELINE_VERSION = 2


MODEL_URI = "gemma4:26b"
OLLAMA_BASE_URL = "http://ollama:11434"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

TOP_K = 40  # retrieve more before rerank
FINAL_K = 10  # chunks kept after reranking

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # to generate vector embeddings for semantic retrieval
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # to rerank retrieved chunks

# Canonical macro-regions used throughout the system.
VALID_REGIONS = {"north", "center", "south", "islands"}

# For a student who is willing to stay relatively close to home,
# define which macro-regions are considered acceptable.
NEARBY_REGIONS = {
    "north": {"north", "center"},
    "center": {"north", "center", "south"},
    "south": {"center", "south", "islands"},
    "islands": {"south", "islands"},
}
