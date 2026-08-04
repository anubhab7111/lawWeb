"""
ISO-639-1 (what fastText emits) → IndicTrans2 language tag (what the
translation models expect, e.g. ``hin_Deva``) mapping.

IndicTrans2 identifies a language by a ``<script>`` FLORES-200-style tag, not
an ISO code, so detection and translation speak different alphabets of
language codes and must be bridged here. Any language not in this table is
treated as unsupported and falls back to English (no translation).
"""

from __future__ import annotations

# ISO-639-1 → IndicTrans2 tag. Covers the 22 scheduled Indian languages
# IndicTrans2 supports (plus Nepali/Sanskrit) — the set named in the spec.
ISO_TO_INDICTRANS2: dict[str, str] = {
    "hi": "hin_Deva",  # Hindi
    "bn": "ben_Beng",  # Bengali
    "ta": "tam_Taml",  # Tamil
    "te": "tel_Telu",  # Telugu
    "kn": "kan_Knda",  # Kannada
    "ml": "mal_Mlym",  # Malayalam
    "mr": "mar_Deva",  # Marathi
    "gu": "guj_Gujr",  # Gujarati
    "pa": "pan_Guru",  # Punjabi (Gurmukhi)
    "ur": "urd_Arab",  # Urdu
    "or": "ory_Orya",  # Odia
    "as": "asm_Beng",  # Assamese
    "sa": "san_Deva",  # Sanskrit
    "kok": "gom_Deva",  # Konkani (Goan)
    "ks": "kas_Arab",  # Kashmiri (Perso-Arabic)
    "mni": "mni_Mtei",  # Manipuri (Meitei)
    "ne": "npi_Deva",  # Nepali
    "sd": "snd_Arab",  # Sindhi
    "mai": "mai_Deva",  # Maithili
    "bho": "bho_Deva",  # Bhojpuri
    "sat": "sat_Olck",  # Santali (Ol Chiki)
    "doi": "doi_Deva",  # Dogri
    "brx": "brx_Deva",  # Bodo
}

# IndicTrans2's tag for English — the pivot language of the whole pipeline.
ENGLISH_TAG = "eng_Latn"


def to_indictrans2_tag(iso_code: str) -> str | None:
    """Return the IndicTrans2 tag for an ISO-639-1 code, or None if the
    language is unsupported (caller should fall back to English)."""
    return ISO_TO_INDICTRANS2.get(iso_code.lower())


def is_supported(iso_code: str) -> bool:
    """Whether we can translate to/from this ISO-639-1 language."""
    return iso_code.lower() in ISO_TO_INDICTRANS2
