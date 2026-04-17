
import os
import uuid
import chromadb
from chromadb.config import Settings
from config import CHROMA_DB_PATH, COLLECTION_NAME, TOP_K, CHROMA_BATCH_SIZE
from logger import get_logger

log = get_logger(__name__)


def _get_collection():
    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


def add_documents(chunks, embeddings, metadatas=None):
    collection = _get_collection()
    total      = len(chunks)

    if metadatas is None:
        metadatas = [{} for _ in chunks]

    for start in range(0, total, CHROMA_BATCH_SIZE):
        end = min(start + CHROMA_BATCH_SIZE, total)
        collection.add(
            documents  = chunks[start:end],
            embeddings = embeddings[start:end],
            metadatas  = metadatas[start:end],
            ids        = [str(uuid.uuid4()) for _ in range(end - start)]
        )
        log.info("Saved batch %d-%d", start, end)

    log.info("Saved %d chunks to vector store", total)


def query_documents(query_embedding, n_results=TOP_K):
    collection = _get_collection()
    count      = collection.count()

    if count == 0:
        return []

    n_results = min(n_results, count)
    results   = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas"]
    )

    output = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        meta = meta or {}
        output.append({
            "text":   doc,
            "source": meta.get("source", "unknown"),
            "page":   meta.get("page", "?")
        })

    return output


def clear_collection():
    chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    ).delete_collection(COLLECTION_NAME)
    log.info("Collection cleared")


def get_storage_stats():
    total_bytes = 0
    if os.path.exists(CHROMA_DB_PATH):
        for dirpath, _, filenames in os.walk(CHROMA_DB_PATH):
            for fname in filenames:
                try:
                    total_bytes += os.path.getsize(os.path.join(dirpath, fname))
                except OSError:
                    pass

    if   total_bytes < 1_024:        readable = f"{total_bytes} B"
    elif total_bytes < 1_024 ** 2:   readable = f"{total_bytes / 1_024:.1f} KB"
    elif total_bytes < 1_024 ** 3:   readable = f"{total_bytes / 1_024 ** 2:.2f} MB"
    else:                             readable = f"{total_bytes / 1_024 ** 3:.2f} GB"

    return {
        "size_readable": readable,
        "chunk_count":   _get_collection().count()
    }
