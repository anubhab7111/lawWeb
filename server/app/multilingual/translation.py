"""
Translation service backed by IndicTrans2 (AI4Bharat), served via CTranslate2.

Two distilled checkpoints are loaded lazily and shared process-wide (the same
singleton-with-lock pattern as the embedding model): ``indic-en`` for
query→English and ``en-indic`` for answer→user-language. Everything runs on CPU
by default so the 4GB VRAM stays reserved for Ollama's LLM.

Why CTranslate2 rather than transformers: this environment runs transformers 5.x
(required by sentence-transformers / BGE-M3 / the reranker), but IndicTrans2's
HuggingFace ``trust_remote_code`` modeling + tokenizer were written for
transformers ~4.x and break on 5.x in several places. CTranslate2 runs the model
with its own inference engine and needs no transformers modeling code at all, so
it stays compatible without downgrading the rest of the stack. We use the
non-gated CTranslate2 conversions of Raj Dabre's rotary IndicTrans2 distilled
200M models (``adalat-ai/ct2-rotary-indictrans2-*``); tokenization is done with
the bundled SentencePiece models plus ``IndicProcessor`` for normalization,
language-tagging and script transliteration.

Design guarantees:
- Legal references are masked (see :mod:`entity_guard`) before translation and
  restored after, so citations survive verbatim (Requirement 3).
- ``translate_batch`` is blocking, so it is offloaded to a thread and the public
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


def _install_transformers_compat_shim() -> None:
    """Make ``IndicTransToolkit`` importable under transformers 5.x.

    IndicTransToolkit's package ``__init__`` imports ``PreTrainedTokenizerBase``
    from ``transformers.tokenization_utils``, but transformers 5.x dropped that
    re-export (it lives at the top level / in ``tokenization_utils_base`` now).
    Re-attaching it lets us keep transformers 5.x for the rest of the stack
    (BGE-M3, reranker, sentence-transformers) instead of pinning an old release.
    We only use IndicProcessor for text normalization — no transformers modeling
    code is loaded, since CTranslate2 runs the model.
    """
    try:
        import transformers.tokenization_utils as _tu

        if not hasattr(_tu, "PreTrainedTokenizerBase"):
            from transformers import PreTrainedTokenizerBase as _ptb

            _tu.PreTrainedTokenizerBase = _ptb  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass


def _cache_key(text: str, src: str, tgt: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return f"{src}>{tgt}:{digest}"


class _Direction:
    """One translation direction: a CTranslate2 translator plus its source/target
    SentencePiece models."""

    __slots__ = ("translator", "sp_src", "sp_tgt")

    def __init__(self, translator: Any, sp_src: Any, sp_tgt: Any) -> None:
        self.translator = translator
        self.sp_src = sp_src
        self.sp_tgt = sp_tgt


class IndicTrans2Service:
    """Lazy-loaded, thread-safe wrapper around the two IndicTrans2 directions."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._loaded = False
        self._load_failed = False
        self._processor: Optional[Any] = None
        self._indic_en: Optional[_Direction] = None
        self._en_indic: Optional[_Direction] = None
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

    @staticmethod
    def _load_direction(repo: str, device: str) -> "_Direction":
        """Download a CTranslate2 IndicTrans2 repo and load its translator + spm.

        The repos nest the model one directory deep
        (``<name>-ct2/ctranslate2_model/{model.bin,config.json,vocab/model.SRC,…}``),
        so we locate the folder holding ``model.bin`` rather than hardcoding it.
        """
        import os

        import ctranslate2
        import sentencepiece as spm
        from huggingface_hub import snapshot_download

        root = snapshot_download(repo)
        model_dir: Optional[str] = None
        for dirpath, _dirs, files in os.walk(root):
            if "model.bin" in files and "config.json" in files:
                model_dir = dirpath
                break
        if model_dir is None:
            raise FileNotFoundError(f"no CTranslate2 model.bin found under {repo!r}")

        translator = ctranslate2.Translator(model_dir, device=device)
        sp_src = spm.SentencePieceProcessor(
            model_file=os.path.join(model_dir, "vocab", "model.SRC")
        )
        sp_tgt = spm.SentencePieceProcessor(
            model_file=os.path.join(model_dir, "vocab", "model.TGT")
        )
        return _Direction(translator, sp_src, sp_tgt)

    def _load_blocking(self) -> None:
        """Download + load both directions. Runs in an executor thread."""
        _install_transformers_compat_shim()

        try:
            from IndicTransToolkit.processor import IndicProcessor
        except ImportError:  # older/newer package layout
            from IndicTransToolkit import IndicProcessor  # type: ignore

        settings = get_settings()
        device = self._resolve_device()
        logger.info("Loading IndicTrans2 via CTranslate2 (device=%s)…", device)

        self._processor = IndicProcessor(inference=True)
        self._indic_en = self._load_direction(
            settings.translation_model_indic_en, device
        )
        self._en_indic = self._load_direction(
            settings.translation_model_en_indic, device
        )
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
        direction = self._indic_en if tgt_tag == ENGLISH_TAG else self._en_indic
        assert direction is not None

        masked, mapping = entity_guard.mask(text)

        # IndicProcessor normalizes and prepends the two flores language tags:
        # "<src_tag> <tgt_tag> <normalized text>". CTranslate2 expects a list of
        # subword tokens, and the tags are atomic vocabulary entries (not part of
        # the SentencePiece model), so keep them intact and SPM-encode only the
        # sentence body.
        preprocessed = self._processor.preprocess_batch(
            [masked], src_lang=src_tag, tgt_lang=tgt_tag
        )[0]
        parts = preprocessed.split(" ")
        tags, body = parts[:2], " ".join(parts[2:])
        tokens = tags + direction.sp_src.encode(body, out_type=str)

        results = direction.translator.translate_batch(
            [tokens],
            beam_size=5,
            max_input_length=256,
            max_decoding_length=_MAX_NEW_TOKENS,
        )
        decoded = direction.sp_tgt.decode(results[0].hypotheses[0])
        out = self._processor.postprocess_batch([decoded], lang=tgt_tag)[0]
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
