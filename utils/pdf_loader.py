
import PyPDF2
from pathlib import Path
from logger import get_logger

log = get_logger(__name__)


def load_pdf(file_path):
    pages = []

    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        total  = len(reader.pages)
        log.info("Opening %s — %d pages", Path(file_path).name, total)

        for i in range(total):
            text = reader.pages[i].extract_text()
            if text and text.strip():
                pages.append({"page": i + 1, "text": text.strip()})

    log.info("Got text from %d / %d pages", len(pages), total)
    return pages
