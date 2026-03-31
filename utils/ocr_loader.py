import pytesseract
from PIL import Image

def load_image_ocr(file_path: str) -> str:
    """
    Extract text from an image using Tesseract OCR.
    Works with PNG, JPG, TIFF, BMP files.
    """
    
    img = Image.open(file_path)

    img = img.convert("L")

    text = pytesseract.image_to_string(img, lang="eng")

    return text.strip()
