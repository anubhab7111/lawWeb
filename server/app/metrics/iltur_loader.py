"""
Loader for the IL-TUR (Indian Legal Text Understanding & Reasoning) benchmark
— specifically its `lsi` (Legal Statute Identification) subtask, used by
tests/test_chatbot.py (default dataset) to sample real case-fact patterns
for evaluating the chatbot's statute-citation accuracy.

IL-TUR is gated on HuggingFace (CC BY-NC-SA 4.0, non-commercial): accept the
license at https://huggingface.co/datasets/Exploration-Lab/IL-TUR and set
HUGGINGFACE_TOKEN in server/.env before using this module.
"""

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
        raise RuntimeError("The `datasets` package is required: pip install datasets") from e

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


def load_iltur_lsi_test_split():
    return _load_iltur_lsi()["test"]


@lru_cache(maxsize=1)
def _statute_section_numbers() -> List[str]:
    """Row i of the `statutes` split is the statute that label index i refers to."""
    return [_section_number(row["id"]) for row in _load_iltur_lsi()["statutes"]]


def sample_iltur_cases(n: int = 20, seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """Randomly sample n rows from the IL-TUR lsi test split.

    seed=None (default) draws a fresh random sample every call. Pass a seed
    for a reproducible sample across runs.
    """
    test_split = load_iltur_lsi_test_split()
    rng = random.Random(seed)
    indices = rng.sample(range(len(test_split)), min(n, len(test_split)))
    return [test_split[i] for i in indices]


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
    """"Section 302" / "Section 120B" / "Section 294(b)" -> "302" / "120B" / "294".

    Sub-clause suffixes are dropped and letter suffixes kept: the index is keyed
    on the base section number.
    """
    match = re.search(r"\d+[A-Za-z]{0,2}", str(label))
    return match.group(0).upper() if match else str(label)


def decode_labels(
    labels: Sequence[Any], names: Optional[Sequence[str]] = None
) -> List[str]:
    """IL-TUR lsi rows carry ClassLabel ints (35 -> "Section 304"); decode them
    to deduplicated section numbers. Already-textual labels pass through."""
    out: List[str] = []
    for label in labels:
        if isinstance(label, int) and not isinstance(label, bool):
            label = (names if names is not None else label_names())[label]
        number = _section_number(label)
        if number not in out:
            out.append(number)
    return out


def iltur_case_to_ground_truth(row: Dict[str, Any], prompt: str) -> GroundTruthEntry:
    """Build a GroundTruthEntry from an IL-TUR lsi row for Hit Rate@k / MRR scoring.

    reference_answer/expected_acts/relevant_keywords are left empty: IL-TUR's
    ground truth is a statute-section label set, not a gold prose answer, so
    only the section-based metrics (Hit Rate@k, MRR) are meaningful here —
    the LLM-judge metrics degrade gracefully for domain="unknown"-style empty
    ground truth, same as any other query MetricsEvaluator can't find a
    reference answer for.
    """
    statutes = _statute_section_numbers()
    # "Section 294" and "Section 294(b)" are separate labels but one section number
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
