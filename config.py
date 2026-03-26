CHROMA_DB_PATH = "./chroma_db"     # Where ChromaDB stores data
COLLECTION_NAME = "documents"       # Collection name in ChromaDB
EMBED_MODEL = "nomic-embed-text"    # Ollama embedding model name
LLM_MODEL = "gemma2:2b"            # Default LLM (can be overridden)
CHUNK_SIZE = 500                     # Characters per chunk
CHUNK_OVERLAP = 50                  # Overlap between chunks
TOP_K = 5                           # How many chunks to retrieve
DATA_DIR = "./data"  