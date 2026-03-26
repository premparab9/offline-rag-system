import re

def clean_text(text: str) -> str:
    """
    Clean raw extracted text.
    Removes noise, normalises whitespace, fixes encoding.
    """
    # Replace Windows-style line endings
    text = text.replace("\r\n", "\n")

    # Remove non-printable characters (except newlines)
    text = re.sub(r'[^\x20-\x7E\n]', " ", text)

    # Collapse multiple spaces into one
    text = re.sub(r' +', " ", text)

    # Collapse more than 2 newlines into 2
    text = re.sub(r'\n{3,}', "\n\n", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text