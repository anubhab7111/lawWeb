"""Loader for the IL-TUR (Indian Legal Text Understanding & Reasoning) benchmark — specifically its `lsi` (Legal Statute Identification) subtask, used by tests/test_chatbot.py..."""

import random
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence

from app.config import get_settings
from app.metrics.ground_truth import GroundTruthEntry

DATASET_ID = "Exploration-Lab/IL-TUR"
DATASET_CONFIG = "lsi"

_PROMPT_TEMPLATE = (
    "Based on the following facts from a legal case in India, which sections "
    "of law would apply, and why?\n\nFacts:\n{facts}"
)


@lru_cache(maxsize=1)
def _load_iltur_lsi():
    """Download (or use the cached copy of) the IL-TUR `lsi` DatasetDict."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "The `datasets` package is required: pip install datasets"
        ) from e

    token = get_settings().huggingface_token or None
    try:
        ds = load_dataset(DATASET_ID, DATASET_CONFIG, token=token)
    except Exception as e:
        raise RuntimeError(
            f"Couldn't load {DATASET_ID}/{DATASET_CONFIG}: {e}. The dataset is gated — "
            "accept its license on HuggingFace and set HUGGINGFACE_TOKEN in server/.env."
        ) from e

    for split in ("test", "statutes"):
        if split not in ds:
            raise RuntimeError(
                f"IL-TUR '{DATASET_CONFIG}' config has no '{split}' split "
                f"(found: {list(ds.keys())})."
            )
    return ds


@lru_cache(maxsize=1)
def _statute_section_numbers() -> List[str]:
    """Row i of the `statutes` split is the statute that label index i refers to."""
    return [_section_number(row["id"]) for row in _load_iltur_lsi()["statutes"]]


def load_iltur_lsi_split(split: str = "test"):
    """One IL-TUR `lsi` split (train / dev / test / statutes)."""
    ds = _load_iltur_lsi()
    if split not in ds:
        raise RuntimeError(
            f"IL-TUR '{DATASET_CONFIG}' config has no '{split}' split "
            f"(found: {list(ds.keys())})."
        )
    return ds[split]


def load_iltur_lsi_test_split():
    return load_iltur_lsi_split("test")


def sample_iltur_cases(
    n: int = 20, seed: Optional[int] = None, split: str = "test"
) -> List[Dict[str, Any]]:
    """Randomly sample n rows from an IL-TUR lsi split (default: test)."""
    rows = load_iltur_lsi_split(split)
    rng = random.Random(seed)
    indices = rng.sample(range(len(rows)), min(n, len(rows)))
    return [rows[i] for i in indices]


def iltur_case_to_prompt(row: Dict[str, Any]) -> str:
    """Turn an IL-TUR lsi row's fact sentences into a chatbot prompt."""
    facts = row["text"]
    if isinstance(facts, list):
        facts = " ".join(facts)
    return _PROMPT_TEMPLATE.format(facts=facts.strip())


@lru_cache(maxsize=1)
def label_names() -> List[str]:
    """IL-TUR lsi label names, indexed by the integer class ids the rows carry."""
    return list(load_iltur_lsi_test_split().features["labels"].feature.names)


def _section_number(label: Any) -> str:
    """ "Section 302" / "Section 120B" / "Section 294(b)" -> "302" / "120B" / "294"."""
    match = re.search(r"\d+[A-Za-z]{0,2}", str(label))
    return match.group(0).upper() if match else str(label)


def decode_labels(
    labels: Sequence[Any], names: Optional[Sequence[str]] = None
) -> List[str]:
    """IL-TUR lsi rows carry ClassLabel ints (35 -> "Section 304"); decode them to deduplicated section numbers."""
    out: List[str] = []
    for label in labels:
        if isinstance(label, int) and not isinstance(label, bool):
            label = (names if names is not None else label_names())[label]
        number = _section_number(label)
        if number not in out:
            out.append(number)
    return out


def iltur_case_to_ground_truth(row: Dict[str, Any], prompt: str) -> GroundTruthEntry:
    """Build a GroundTruthEntry from an IL-TUR lsi row for Hit Rate@k / MRR scoring."""
    statutes = _statute_section_numbers()

    sections = list(dict.fromkeys(statutes[i] for i in row["labels"]))
    return GroundTruthEntry(
        query=prompt,
        relevant_ipc_sections=[],
        relevant_sections=sections,
        relevant_keywords=[],
        expected_acts=[],
        reference_answer="",
        domain="il_tur_lsi",
    )
