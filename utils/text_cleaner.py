
import re

def clean_text(text):
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
