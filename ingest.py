from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP
from pathlib import Path
from utils.pdf_loader import load_pdf
from utils.docx_loader import load_docx
from utils.ocr_loader import load_image_ocr
from utils.text_cleaner import clean_text
from embeddings import embed_texts
from vector_store import add_documents

from tqdm import tqdm   # 🔥 NEW


def load_document(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".docx":
        return load_docx(file_path)
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    elif ext in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}:
        return load_image_ocr(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def ingest_document(file_path: str) -> int:
    print(f"Ingesting: {file_path}")

    # 1. Load
    raw_text = load_document(file_path)

    if not raw_text.strip():
        print("No text found")
        return 0

    # 2. Clean
    cleaned = clean_text(raw_text)

    # 3. Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_text(cleaned)
    print(f"Total chunks: {len(chunks)}")

    # 4. Embedding with tqdm 🔥
    embeddings = []
    for chunk in tqdm(chunks, desc="Embedding chunks"):
        emb = embed_texts([chunk])[0]
        embeddings.append(emb)

    # 5. Store
    add_documents(chunks, embeddings)

    print("Ingestion complete")
    return len(chunks)