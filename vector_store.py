import chromadb
from chromadb.config import Settings
from config import CHROMA_DB_PATH, COLLECTION_NAME, TOP_K
import uuid

def get_chroma_client():
    """
    Create a persistent ChromaDB client.
    Data is saved to disk at CHROMA_DB_PATH.
    """
    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    return client


def get_or_create_collection():
    """
    Get existing collection or create a new one.
    Collections are like tables in a relational DB.
    We use cosine similarity as our distance metric.
    """
    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # Use cosine similarity
    )
    return collection


def add_documents(chunks: list[str], embeddings: list):
    """
    Store text chunks and their embeddings in ChromaDB.

    Args:
        chunks: list of text strings
        embeddings: list of corresponding float vectors
    """
    collection = get_or_create_collection()

    # Generate unique IDs for each chunk
    ids = [str(uuid.uuid4()) for _ in chunks]

    collection.add(
        documents=chunks,      # The actual text (stored for retrieval)
        embeddings=embeddings, # The vector representations
        ids=ids                # Unique identifiers
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB")


def query_documents(query_embedding: list, n_results: int = TOP_K) -> list[str]:
    """
    Retrieve the top-k most similar chunks.

    Args:
        query_embedding: embedding vector of the user's question
        n_results: number of chunks to return

    Returns:
        List of relevant text chunks
    """
    collection = get_or_create_collection()

    results = collection.query(
        query_embeddings=[query_embedding],  # Must be a list of lists
        n_results=n_results,
        include=["documents"]               # Return text, not just IDs
    )

    # results["documents"] is a list of lists — flatten it
    return results["documents"][0]


def clear_collection():
    """Delete all documents from the collection."""
    client = get_chroma_client()
    client.delete_collection(COLLECTION_NAME)
    print("Collection cleared.")