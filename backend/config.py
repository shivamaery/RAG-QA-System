from pathlib import Path
import os, logging

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "test_data"
CHROMA_DB_DIR = BASE_DIR / "db" / "chroma"
CHROMA_COLLECTION_NAME = "dal_theses"

# Model / LLM / embedding settings
MODEL_NAME = os.getenv("MODEL_NAME", "microsoft/Phi-4-mini-instruct")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")

# Chunking (optimized for retrieval with Phi-3 Mini)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))     
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150)) 

# Generation config
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.4))
TOP_P = float(os.getenv("TOP_P", 0.9))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", 1.15))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 512))

# Chroma / persistence
CHROMA_PERSIST_DIR = str(CHROMA_DB_DIR)

# Retrieval
NUM_RETRIEVE = int(os.getenv("NUM_RETRIEVE", 7))

# Deterministic Results
DO_SAMPLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(f"Using Model: {MODEL_NAME}, Embedding: {EMBEDDING_MODEL}")
logger.info(f"Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")
