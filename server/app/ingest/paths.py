"""
Where corpus data lives.

Raw and derived corpus data (PDFs, judgments, chunks, caches) is kept on an
external drive, `LAWWEB_CORPUS_ROOT`; only vector/lexical indices live in the
repo. The drive is removable, so nothing that serves queries may depend on it —
build/ingest code calls `require_corpus()`, serving code just loads the index.
With no corpus root configured, the legacy in-repo layout (`app/data/...`) is
used, which keeps a fresh clone working until its data is moved.
"""

from pathlib import Path
from typing import Optional

from app.config import get_settings

LEGACY_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class CorpusUnavailable(RuntimeError):
    pass


def corpus_root() -> Optional[Path]:
    root = get_settings().corpus_root.strip()
    return Path(root).expanduser() if root else None


def corpus_available() -> bool:
    root = corpus_root()
    return root is not None and root.is_dir()


def require_corpus() -> Path:
    root = corpus_root()
    if root is None:
        raise CorpusUnavailable(
            "LAWWEB_CORPUS_ROOT is not set; set it in server/.env to the corpus drive."
        )
    if not root.is_dir():
        raise CorpusUnavailable(
            f"Corpus root {root} is not available — is the drive mounted?"
        )
    return root


def corpus_path(*parts: str) -> Path:
    return require_corpus().joinpath(*parts)


def statutes_dir() -> Path:
    """Directory holding bare_acts/ and the prose corpora (mappings, rules, ...)."""
    root = corpus_root()
    if root is not None and (root / "statutes").is_dir():
        return root / "statutes"
    return LEGACY_DATA_DIR


def case_law_dir() -> Path:
    """Curated landmark-judgment JSON files."""
    root = corpus_root()
    if root is not None and (root / "case_law" / "curated").is_dir():
        return root / "case_law" / "curated"
    return LEGACY_DATA_DIR / "case_law"
