"""Translation service backed by IndicTrans2 (AI4Bharat), served via CTranslate2."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from typing import Any, Optional

from cachetools import TTLCache

from app.config import get_settings

from . import entity_guard
from .lang_map import ENGLISH_TAG

logger = logging.getLogger(__name__)


_MAX_NEW_TOKENS = 512


def _install_transformers_compat_shim() -> None:
    """Make ``IndicTransToolkit`` importable under transformers 5.x."""
    try:
        import transformers.tokenization_utils as _tu

        if not hasattr(_tu, "PreTrainedTokenizerBase"):
            from transformers import PreTrainedTokenizerBase as _ptb

            _tu.PreTrainedTokenizerBase = _ptb  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass


_SENTENCE_END_RE = re.compile(r"(?<=[.!?।])\s+")
_MAX_SEGMENT_CHARS = 400


def split_for_translation(line: str, limit: int = _MAX_SEGMENT_CHARS) -> list[str]:
    """Split one line into pieces the translator can take whole, on sentence ends first and on word boundaries for a sentence that is itself too long."""
    if len(line) <= limit:
        return [line]
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_END_RE.split(line):
        while len(sentence) > limit:
            cut = sentence.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:cut])
            sentence = sentence[cut:].lstrip()
        if current and len(current) + 1 + len(sentence) > limit:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def _cache_key(text: str, src: str, tgt: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return f"{src}>{tgt}:{digest}"


class _Direction:
    """One translation direction: a CTranslate2 translator plus its source/target SentencePiece models."""

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


    def _resolve_device(self) -> str:
        device = get_settings().translation_device
        if device in ("cuda", "cpu"):
            return device


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
        """Download a CTranslate2 IndicTrans2 repo and load its translator + spm."""
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
        """Download + load both directions."""
        _install_transformers_compat_shim()

        try:
            from IndicTransToolkit.processor import IndicProcessor
        except ImportError:
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


    def _translate_blocking(self, text: str, src_tag: str, tgt_tag: str) -> str:
        assert self._processor is not None
        direction = self._indic_en if tgt_tag == ENGLISH_TAG else self._en_indic
        assert direction is not None


        lines = text.split("\n")
        segments: list[tuple[int, str]] = []
        for i, line in enumerate(lines):
            if line.strip():
                segments.extend((i, part) for part in split_for_translation(line))
        if not segments:
            return text

        masked_parts: list[str] = []
        mappings: list[dict[str, str]] = []
        for _, part in segments:
            masked, mapping = entity_guard.mask(part)
            masked_parts.append(masked)
            mappings.append(mapping)


        preprocessed = self._processor.preprocess_batch(
            masked_parts, src_lang=src_tag, tgt_lang=tgt_tag
        )
        batch = []
        for item in preprocessed:
            parts = item.split(" ")
            batch.append(parts[:2] + direction.sp_src.encode(" ".join(parts[2:]), out_type=str))

        results = direction.translator.translate_batch(
            batch,
            beam_size=5,
            max_input_length=512,
            max_decoding_length=_MAX_NEW_TOKENS,
        )
        decoded = [direction.sp_tgt.decode(r.hypotheses[0]) for r in results]
        outputs = self._processor.postprocess_batch(decoded, lang=tgt_tag)

        rebuilt: dict[int, list[str]] = {}
        for (line_no, _), out, mapping in zip(segments, outputs, mappings):
            rebuilt.setdefault(line_no, []).append(entity_guard.unmask(out, mapping))
        return "\n".join(
            " ".join(rebuilt[i]) if i in rebuilt else line for i, line in enumerate(lines)
        )

    async def translate(self, text: str, src_tag: str, tgt_tag: str) -> str:
        """Translate ``text`` from ``src_tag`` to ``tgt_tag`` (IndicTrans2 tags)."""
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
    """Return the process-wide translation service (constructed on first use; models still load lazily inside it)."""
    global _service
    if _service is None:
        _service = IndicTrans2Service()
    return _service
