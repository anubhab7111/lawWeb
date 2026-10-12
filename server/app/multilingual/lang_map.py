"""ISO-639-1 (what fastText emits) → IndicTrans2 language tag (what the translation models expect, e.g."""

from __future__ import annotations

ISO_TO_INDICTRANS2: dict[str, str] = {
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "pa": "pan_Guru",
    "ur": "urd_Arab",
    "or": "ory_Orya",
    "as": "asm_Beng",
    "sa": "san_Deva",
    "kok": "gom_Deva",
    "ks": "kas_Arab",
    "mni": "mni_Mtei",
    "ne": "npi_Deva",
    "sd": "snd_Arab",
    "mai": "mai_Deva",
    "bho": "bho_Deva",
    "sat": "sat_Olck",
    "doi": "doi_Deva",
    "brx": "brx_Deva",
}


ENGLISH_TAG = "eng_Latn"


def to_indictrans2_tag(iso_code: str) -> str | None:
    """Return the IndicTrans2 tag for an ISO-639-1 code, or None if the language is unsupported (caller should fall back to English)."""
    return ISO_TO_INDICTRANS2.get(iso_code.lower())
