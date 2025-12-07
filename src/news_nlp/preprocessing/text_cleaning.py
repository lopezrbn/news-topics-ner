import re
from typing import Optional


def clean_text(text: Optional[str]) -> str:
    """
    Clean a single text string.

    Current implementation is a simple placeholder:
      - convert to string
      - strip leading/trailing whitespace
      - lowercase
      - collapse multiple spaces

    Parameters
    ----------
    text : Optional[str]
        Raw input text.

    Returns
    -------
    cleaned : str
        Cleaned text.
    """
    if text is None:
        return ""

    text = str(text)  # Ensure it's a string
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'\n', ' ', text)   # Remove new line characters
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = text.strip()  # Strip leading/trailing whitespace

    return text
