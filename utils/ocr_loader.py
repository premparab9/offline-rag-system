import pytesseract
from PIL import Image

def load_image_ocr(file_path: str) -> str:
    """
    Extract text from an image using Tesseract OCR.
    Works with PNG, JPG, TIFF, BMP files.
    """
    # Open the image with Pillow
    img = Image.open(file_path)

    # Convert to grayscale for better OCR accuracy
    img = img.convert("L")

    # Run Tesseract on the image
    # lang='eng' specifies English; add '+fra' for French etc.
    text = pytesseract.image_to_string(img, lang="eng")

    return text.strip()