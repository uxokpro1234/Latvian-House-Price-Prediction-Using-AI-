
import unicodedata

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return str(text) if text is not None else ""

    text = text.lower().strip()

    replacements = {
        "ā": "a", "č": "c", "ē": "e", "ģ": "g", "ī": "i",
        "ķ": "k", "ļ": "l", "ņ": "n", "š": "s", "ū": "u", "ž": "z"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    return text
