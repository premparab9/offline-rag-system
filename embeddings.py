from langchain_community.embeddings import OllamaEmbeddings
from config import EMBED_MODEL

def get_embedding_model():
    """
    Return the nomic-embed-text model via Ollama.
    This model generates 768-dimensional vectors.
    Ollama must be running (ollama serve).
    """
    return OllamaEmbeddings(model=EMBED_MODEL)


def embed_texts(texts: list) -> list:
    """
    Generate embeddings for a list of text chunks.
    Returns a list of 768-dimensional float vectors.
    """
    model = get_embedding_model()

    
    embeddings = model.embed_documents(texts)
    return embeddings


def embed_query(query: str) -> list:
    """
    Generate embedding for a single query string.
    Uses embed_query() which is optimised for single strings.
    """
    model = get_embedding_model()
    return model.embed_query(query)
