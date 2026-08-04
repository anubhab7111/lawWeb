"""
Translation service backed by IndicTrans2 (AI4Bharat).

Two distilled checkpoints are loaded lazily and shared process-wide (the same
singleton-with-lock pattern as the embedding model): ``indic-en`` for
query→English and ``en-indic`` for answer→user-language. Everything runs on CPU
by default so the 4GB VRAM stays reserved for Ollama's LLM.

Design guarantees:
- Legal references are masked (see :mod:`entity_guard`) before translation and
  restored after, so citations survive verbatim (Requirement 3).
- ``model.generate`` is blocking, so it is offloaded to a thread and the public
  API is async (Requirement 7).
- Results are cached in a TTLCache keyed on (src, tgt, text) when enabled.
- Any failure returns the *input text unchanged* and logs — the conversation
  continues in English rather than breaking (Requirement 9).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any, Optional

from cachetools import TTLCache

from app.config import get_settings

from . import entity_guard
from .lang_map import ENGLISH_TAG

logger = logging.getLogger(__name__)

# Generation cap — legal answers are bounded; keeps CPU latency predictable.
_MAX_NEW_TOKENS = 512


def _cache_key(text: str, src: str, tgt: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return f"{src}>{tgt}:{digest}"


class IndicTrans2Service:
    """Lazy-loaded, thread-safe wrapper around the two IndicTrans2 directions."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._loaded = False
        self._load_failed = False
        self._processor: Optional[Any] = None
        self._indic_en: Optional[tuple[Any, Any]] = None  # (tokenizer, model)
        self._en_indic: Optional[tuple[Any, Any]] = None
        settings = get_settings()
        self._cache: Optional[TTLCache] = (
            TTLCache(maxsize=1024, ttl=settings.cache_ttl_seconds)
            if settings.translation_cache
            else None
        )

    # ------------------------------------------------------------------ load
    def _resolve_device(self) -> str:
        device = get_settings().translation_device
        if device in ("cuda", "cpu"):
            return device
        # "auto": only take the GPU when it is comfortably large; on the 4GB
        # target card the VRAM is worth far more to Ollama, so default to CPU.
        try:
            import torch

            if torch.cuda.is_available():
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                if total_bytes >= 8 * 1024**3 and free_bytes > 4 * 1024**3:
                    return "cuda"
        except Exception:  # noqa: BLE001
            pass
        return "cpu"

    def _load_blocking(self) -> None:
        """Import + load both checkpoints. Runs in an executor thread."""
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        try:
            from IndicTransToolkit.processor import IndicProcessor
        except ImportError:  # older/newer package layout
            from IndicTransToolkit import IndicProcessor  # type: ignore

        settings = get_settings()
        device = self._resolve_device()
        logger.info("Loading IndicTrans2 (device=%s)…", device)

        def _load_pair(name: str) -> tuple[Any, Any]:
            tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                name, trust_remote_code=True
            ).to(device)
            model.eval()
            return tok, model

        self._processor = IndicProcessor(inference=True)
        self._indic_en = _load_pair(settings.translation_model_indic_en)
        self._en_indic = _load_pair(settings.translation_model_en_indic)
        self._device = device
        logger.info("IndicTrans2 ready.")

    async def _ensure_loaded(self) -> bool:
        if self._loaded:
            return True
        if self._load_failed:
            return False
        async with self._lock:
            if self._loaded:
                return True
            if self._load_failed:
                return False
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._load_blocking)
                self._loaded = True
            except Exception as exc:  # noqa: BLE001
                self._load_failed = True
                logger.warning("IndicTrans2 unavailable (%s); staying English-only", exc)
            return self._loaded

    # ------------------------------------------------------------- translate
    def _translate_blocking(self, text: str, src_tag: str, tgt_tag: str) -> str:
        assert self._processor is not None
        pair = self._indic_en if tgt_tag == ENGLISH_TAG else self._en_indic
        assert pair is not None
        tokenizer, model = pair

        masked, mapping = entity_guard.mask(text)
        batch = self._processor.preprocess_batch(
            [masked], src_lang=src_tag, tgt_lang=tgt_tag
        )
        import torch

        inputs = tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            max_length=1024,
        ).to(self._device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                num_beams=5,
                max_new_tokens=_MAX_NEW_TOKENS,
                num_return_sequences=1,
            )
        decoded = tokenizer.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        out = self._processor.postprocess_batch(decoded, lang=tgt_tag)[0]
        return entity_guard.unmask(out, mapping)

    async def translate(self, text: str, src_tag: str, tgt_tag: str) -> str:
        """Translate ``text`` from ``src_tag`` to ``tgt_tag`` (IndicTrans2 tags).

        Returns the input unchanged on any failure (never raises). No-op when
        source and target tags are identical.
        """
        if not text or not text.strip() or src_tag == tgt_tag:
            return text

        if self._cache is not None:
            key = _cache_key(text, src_tag, tgt_tag)
            cached = self._cache.get(key)
            if cached is not None:
                return cached

        if not await self._ensure_loaded():
            return text

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._translate_blocking, text, src_tag, tgt_tag
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Translation %s→%s failed (%s); returning source", src_tag, tgt_tag, exc)
            return text

        if self._cache is not None:
            self._cache[_cache_key(text, src_tag, tgt_tag)] = result
        return result


_service: Optional[IndicTrans2Service] = None


def get_translation_service() -> IndicTrans2Service:
    """Return the process-wide translation service (constructed on first use;
    models still load lazily inside it)."""
    global _service
    if _service is None:
        _service = IndicTrans2Service()
    return _service
