"""Word-boundary keyword matching shared by the keyword-driven classifiers.

Plain `term in text` matches inside longer words ("fire" in "fired", "fir" in
"first"). These helpers match whole words, allowing only regular inflections
of the last word ("kill" -> killed/killing/kills, "stab" -> stabbed).
"""

import re
from functools import lru_cache
from typing import Iterable


@lru_cache(maxsize=4096)
def _pattern(term: str) -> "re.Pattern[str]":
    term = term.lower()
    last = re.escape(term[-1])
    inflect = rf"(?:s|es|ed|ing|{last}ed|{last}ing)?"
    return re.compile(r"(?<![a-z0-9])" + re.escape(term) + inflect + r"(?![a-z0-9])")


def contains_word(text: str, term: str) -> bool:
    return bool(_pattern(term).search(text.lower()))


def any_word(text: str, terms: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(_pattern(t).search(lowered) for t in terms)


def count_words(text: str, terms: Iterable[str]) -> int:
    lowered = text.lower()
    return sum(1 for t in terms if _pattern(t).search(lowered))
