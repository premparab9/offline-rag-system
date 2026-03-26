from docx import Document

def load_docx(file_path: str) -> str:
    """
    Extract text from a .docx Word file.
    Preserves paragraph structure.
    """
    doc = Document(file_path)
    paragraphs = []

    for para in doc.paragraphs:
        # Skip empty paragraphs
        if para.text.strip():
            paragraphs.append(para.text.strip())

    return "\n".join(paragraphs)