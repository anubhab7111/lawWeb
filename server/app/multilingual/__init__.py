"""Multilingual layer for the LawWeb chatbot."""

from .language_detection import LanguageResult, detect_language
from .pipeline import postprocess_response, preprocess_query
from .translation import get_translation_service

__all__ = [
    "LanguageResult",
    "detect_language",
    "preprocess_query",
    "postprocess_response",
    "get_translation_service",
]
