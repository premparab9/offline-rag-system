
import pytesseract
from PIL import Image
from logger import get_logger

log = get_logger(__name__)


def load_image_ocr(file_path):
    img  = Image.open(file_path).convert("L")
    text = pytesseract.image_to_string(img, lang="eng").strip()

    if not text:
        log.warning("No text found in image: %s", file_path)
        return []

    log.info("OCR done: %s", file_path)
    return [{"page": "image", "text": text}]
