"""
Offline language detection via fastText's ``lid.176.bin`` model.

The model is loaded once and shared (module-global + lock, mirroring the
embedding singleton in ``tools/base_legal_rag.py``). Detection never raises:
any failure — missing model file, load error, empty input — degrades to the
configured default language (English) so the conversation is never interrupted
(Requirement 9).
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LanguageResult:
    """Outcome of language detection.

    Attributes:
        language: ISO-639-1 code (e.g. ``"hi"``); the default language on any
            failure or low-confidence guess.
        confidence: fastText probability for the top label (0.0 when defaulted).
        is_reliable: True only when a language other than the default was
            detected at or above the confidence threshold. Callers use this to
            decide whether translation is warranted.
    """

    language: str
    confidence: float
    is_reliable: bool


_model: Optional[Any] = None
_model_lock = asyncio.Lock()
_load_failed = False


def _default_result() -> LanguageResult:
    return LanguageResult(
        language=get_settings().default_language,
        confidence=0.0,
        is_reliable=False,
    )


async def _get_model() -> Optional[Any]:
    """Lazily load the fastText model once. Returns None if it can't be loaded
    (missing file or import error) — detection then defaults to English."""
    global _model, _load_failed
    if _model is not None or _load_failed:
        return _model
    async with _model_lock:
        if _model is not None or _load_failed:
            return _model
        settings = get_settings()
        path = settings.lang_detect_model_path
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"fastText model not found at {path!r} — download lid.176.bin "
                    "(see docs/multilingual.md)"
                )
            import fasttext

            loop = asyncio.get_event_loop()
            # fastText prints a harmless load warning to stderr; suppressed by
            # loading in an executor without capturing.
            _model = await loop.run_in_executor(
                None, lambda: fasttext.load_model(path)
            )
            logger.info("Loaded fastText language detector from %s", path)
        except Exception as exc:  # noqa: BLE001 — must never propagate
            _load_failed = True
            logger.warning(
                "Language detection unavailable (%s); defaulting to %s",
                exc,
                settings.default_language,
            )
        return _model


async def detect_language(text: str) -> LanguageResult:
    """Detect the language of ``text``.

    Returns the default language (English) with ``is_reliable=False`` when the
    detector is unavailable, the input is blank, the top language *is* the
    default, or confidence is below ``lang_detect_min_confidence``.
    """
    settings = get_settings()
    default_lang = settings.default_language

    stripped = (text or "").strip()
    if not stripped:
        return _default_result()

    model = await _get_model()
    if model is None:
        return _default_result()

    try:
        # fastText chokes on newlines; feed a single flattened line.
        sample = " ".join(stripped.split())
        # Call the underlying pybind predictor directly: fastText's Python
        # ``predict`` wrapper wraps probabilities with ``np.array(probs,
        # copy=False)``, which NumPy 2.x rejects (it can't avoid the copy),
        # making every detection fail. ``model.f.predict`` returns raw
        # ``[(prob, label), …]`` tuples and sidesteps NumPy entirely.
        predictions = model.f.predict(sample, 1, 0.0, "strict")
    except Exception as exc:  # noqa: BLE001
        logger.warning("fastText prediction failed (%s); defaulting to %s", exc, default_lang)
        return _default_result()

    if not predictions:
        return _default_result()

    # Each prediction is (probability, "__label__hi").
    confidence, label = predictions[0]
    iso = label.replace("__label__", "")
    confidence = float(confidence)

    if iso == default_lang:
        return LanguageResult(language=default_lang, confidence=confidence, is_reliable=False)

    if confidence < settings.lang_detect_min_confidence:
        logger.debug(
            "Low-confidence detection %s (%.2f < %.2f); defaulting to %s",
            iso, confidence, settings.lang_detect_min_confidence, default_lang,
        )
        return LanguageResult(language=default_lang, confidence=confidence, is_reliable=False)

    return LanguageResult(language=iso, confidence=confidence, is_reliable=True)
