"""
Detect numbers in text (both digit and text format).
"""

import re
from text_to_num import text2num
from ..utils.constants import DEFAULT_LANGUAGE


def detect_number_in_text(text: str, lang: str = DEFAULT_LANGUAGE) -> bool:
    """
    Check if text contains any number in digit or text format.

    Args:
        text: Input text to check
        lang: Language code for text-to-number conversion

    Returns:
        bool: True if text contains numbers (digit or text), False otherwise
    """
    if not text:
        return False

    # Check for digit numbers using regex
    if re.search(r"\d+", text):
        return True

    # Check for text numbers using text2num
    words = text.lower().split()
    for word in words:
        clean_word = re.sub(r"[^\w\s]", "", word)
        try:
            text2num(clean_word, lang)
            return True
        except ValueError:
            pass

    return False
