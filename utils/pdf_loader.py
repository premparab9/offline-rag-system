import PyPDF2

def load_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file
    """
    text = ""

    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    return text.strip()