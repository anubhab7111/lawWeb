"""
Loader for the IL-TUR (Indian Legal Text Understanding & Reasoning) benchmark
— specifically its `lsi` (Legal Statute Identification) subtask, used by
tests/test_chatbot_iltur.py to sample real case-fact patterns for evaluating
the chatbot's statute-citation accuracy.

IL-TUR is gated on HuggingFace (CC BY-NC-SA 4.0, non-commercial): accept the
license at https://huggingface.co/datasets/Exploration-Lab/IL-TUR and set
HUGGINGFACE_TOKEN in server/.env before using this module.
"""

import random
from typing import Any, Dict, List, Optional

from app.config import get_settings
from app.metrics.ground_truth import GroundTruthEntry

DATASET_ID = "Exploration-Lab/IL-TUR"
DATASET_CONFIG = "lsi"

_PROMPT_TEMPLATE = (
    "Based on the following facts from a legal case in India, which sections "
    "of law would apply, and why?\n\nFacts:\n{facts}"
)


def load_iltur_lsi_test_split():
    """Download (or use the cached copy of) the IL-TUR `lsi` test split."""
    from datasets import load_dataset

    token = get_settings().huggingface_token or None
    try:
        ds = load_dataset(DATASET_ID, DATASET_CONFIG, token=token)
    except Exception:
        # Some HF dataset-viewer conversions require the loading-script
        # revision explicitly rather than the auto-converted parquet default.
        ds = load_dataset(
            DATASET_ID,
            DATASET_CONFIG,
            revision="script",
            trust_remote_code=True,
            token=token,
        )

    if "test" not in ds:
        raise RuntimeError(
            f"IL-TUR '{DATASET_CONFIG}' config has no 'test' split "
            f"(found: {list(ds.keys())})."
        )
    return ds["test"]


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


def iltur_case_to_ground_truth(row: Dict[str, Any], prompt: str) -> GroundTruthEntry:
    """Build a GroundTruthEntry from an IL-TUR lsi row for Hit Rate@k / MRR scoring.

    reference_answer/expected_acts/relevant_keywords are left empty: IL-TUR's
    ground truth is a statute-section label set, not a gold prose answer, so
    only the section-based metrics (Hit Rate@k, MRR) are meaningful here —
    the LLM-judge metrics degrade gracefully for domain="unknown"-style empty
    ground truth, same as any other query MetricsEvaluator can't find a
    reference answer for.
    """
    sections = [str(s) for s in row["labels"]]
    return GroundTruthEntry(
        query=prompt,
        relevant_ipc_sections=[],
        relevant_sections=sections,
        relevant_keywords=[],
        expected_acts=[],
        reference_answer="",
        domain="il_tur_lsi",
    )
