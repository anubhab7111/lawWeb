"""
Multilingual pre/post-processing the chatbot calls.

The chatbot only ever needs two functions:

- :func:`preprocess_query` — detect the input language and, if it is a
  supported non-English language, translate the query to English so the rest of
  the (unchanged) English RAG/Qwen pipeline runs as-is.
- :func:`postprocess_response` — translate the English answer back into the
  user's language for display.

Both are hard no-ops (instant, no model load) when ``multilingual_enabled`` is
False or the language is English/unsupported, preserving full backward
compatibility and adding zero latency to English requests (Requirement 7).
"""

from __future__ import annotations

import logging

from app.config import get_settings

from .lang_map import ENGLISH_TAG, to_indictrans2_tag
from .language_detection import LanguageResult, detect_language
from .translation import get_translation_service

logger = logging.getLogger(__name__)


def _english_result() -> LanguageResult:
    return LanguageResult(
        language=get_settings().default_language, confidence=1.0, is_reliable=False
    )


async def preprocess_query(text: str) -> tuple[str, LanguageResult]:
    """Detect language and translate the query to English when needed.

    Returns ``(english_text, language_result)``. On English input, unsupported
    languages, low confidence, or any failure, returns the original text and a
    non-reliable English result — retrieval and reasoning then proceed exactly
    as in the English-only pipeline.
    """
    settings = get_settings()
    if not settings.multilingual_enabled:
        return text, _english_result()

    lang = await detect_language(text)
    if not lang.is_reliable:
        return text, lang

    tgt_iso = lang.language
    src_tag = to_indictrans2_tag(tgt_iso)
    if src_tag is None:
        # Detected but not a language we translate — carry on in-language;
        # BGE-M3 can still embed it, and Qwen is multilingual enough to cope.
        logger.info("Unsupported language %s; skipping translation", tgt_iso)
        return text, LanguageResult(tgt_iso, lang.confidence, False)

    english = await get_translation_service().translate(text, src_tag, ENGLISH_TAG)
    logger.info("Translated query %s→en (conf=%.2f)", tgt_iso, lang.confidence)
    return english, lang


async def postprocess_response(english_text: str, lang: LanguageResult) -> str:
    """Translate the English answer back into the user's language.

    No-op (returns ``english_text``) when multilingual is disabled, the user's
    language is English, detection was unreliable, or the language is
    unsupported. Never raises.
    """
    if not english_text:
        return english_text

    settings = get_settings()
    if not settings.multilingual_enabled or not lang.is_reliable:
        return english_text
    if lang.language == settings.default_language:
        return english_text

    tgt_tag = to_indictrans2_tag(lang.language)
    if tgt_tag is None:
        return english_text

    translated = await get_translation_service().translate(
        english_text, ENGLISH_TAG, tgt_tag
    )
    logger.info("Translated response en→%s", lang.language)
    return translated
