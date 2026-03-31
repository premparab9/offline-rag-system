import re

def clean_text(text: str) -> str:
    """
    Clean raw extracted text.
    Removes noise, normalises whitespace, fixes encoding.
    """
    
    text = text.replace("\r\n", "\n")
    
    text = re.sub(r'[^\x20-\x7E\n]', " ", text)

    text = re.sub(r' +', " ", text)

    text = re.sub(r'\n{3,}', "\n\n", text)
    text = text.strip()

    return text
