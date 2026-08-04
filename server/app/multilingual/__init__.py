"""
Multilingual layer for the LawWeb chatbot.

Wraps language detection (fastText) and translation (IndicTrans2) around the
existing English RAG/Qwen pipeline: a user's query is translated to English
before retrieval and the English answer is translated back afterward, while
conversation memory stays canonical-English. The whole layer is a no-op when
``settings.multilingual_enabled`` is False.

Public surface (import from here, not the submodules):
    detect_language, LanguageResult, preprocess_query, postprocess_response,
    get_translation_service
"""

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
