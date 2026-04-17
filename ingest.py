
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBED_WORKERS
from embeddings import get_embedding_model
from vector_store import add_documents
from utils.pdf_loader import load_pdf
from utils.docx_loader import load_docx
from utils.ocr_loader import load_image_ocr
from utils.text_cleaner import clean_text
from logger import get_logger

log = get_logger(__name__)

_thread_local = threading.local()

def _get_thread_model():
    if not hasattr(_thread_local, "model"):
        _thread_local.model = get_embedding_model()
    return _thread_local.model

def _embed_chunk(chunk):
    return _get_thread_model().embed_query(chunk)


def load_document(file_path):
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".docx":
        return load_docx(file_path)
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return [{"page": "txt", "text": f.read()}]
    elif ext in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}:
        return load_image_ocr(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def ingest_document(file_path):
    file_name = Path(file_path).name
    log.info("Starting: %s", file_name)

    try:
        pages = load_document(file_path)
    except Exception as e:
        log.error("Failed to load %s: %s", file_name, e)
        return 0

    if not pages:
        log.warning("No text found in %s", file_name)
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    all_chunks   = []
    all_metadata = []

    for page in pages:
        cleaned = clean_text(page["text"])
        if not cleaned:
            continue
        for chunk in splitter.split_text(cleaned):
            all_chunks.append(chunk)
            all_metadata.append({"source": file_name, "page": str(page["page"])})

    total_chunks = len(all_chunks)
    log.info("Split into %d chunks", total_chunks)

    if total_chunks == 0:
        log.warning("No chunks generated from %s", file_name)
        return 0

    all_embeddings = [None] * total_chunks

    with tqdm(total=len(pages) + total_chunks, desc=f"Ingesting {file_name}", unit="step", initial=len(pages)) as bar:
        with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as pool:
            future_map = {pool.submit(_embed_chunk, chunk): idx for idx, chunk in enumerate(all_chunks)}

            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    all_embeddings[idx] = future.result()
                except Exception as e:
                    log.error("Embedding failed for chunk %d: %s", idx, e)
                    return 0
                bar.update(1)

    log.info("Saving to vector store...")
    add_documents(all_chunks, all_embeddings, all_metadata)

    log.info("Done. %d chunks stored from %s", total_chunks, file_name)
    return total_chunks
