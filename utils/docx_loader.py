
from docx import Document
from logger import get_logger

log = get_logger(__name__)


def load_docx(file_path):
    doc  = Document(file_path)
    text = "\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())

    if not text:
        log.warning("No text found in %s", file_path)
        return []

    log.info("Loaded %s", file_path)
    return [{"page": "docx", "text": text}]
