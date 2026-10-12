"""fact_statutes.py — statute identification from a long fact narrative."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from app.metrics.iltur_eval import fact_windows

MAPS_PATH = Path(__file__).resolve().parent.parent / "data" / "section_maps.json"

IPC = "Indian Penal Code"
CRPC = "Code of Criminal Procedure"
BNS = "Bharatiya Nyaya Sanhita BNS"
ACT_WEIGHTS: Dict[str, float] = {IPC: 1.0, CRPC: 0.1, BNS: 0.25}

WINDOWS = 12
DEPTH = 4
RRF_K = 5
CANDIDATE_POOL = 100
RERANK_POOL = 60
RETRIEVAL_K = 60

_maps: Optional[Dict] = None


def _translation() -> Dict[str, Dict[str, List[str]]]:
    global _maps
    if _maps is None:
        _maps = json.loads(MAPS_PATH.read_text())
    return _maps


def numbers_for(act: str, section: str) -> List[str]:
    """Bare label numbers a retrieved provision counts toward."""
    if act == BNS:
        return _translation()["ipc_bns"]["new_to_old"].get(section, [])
    if act in (IPC, CRPC):
        return [section]
    return []


def aggregate(
    per_window: Sequence[Sequence[Tuple[str, str]]],
    weights: Optional[Dict[str, float]] = None,
    depth: int = DEPTH,
    rrf_k: int = RRF_K,
) -> List[Tuple[str, float]]:
    """Combine per-window (act, section) rankings into ranked (number, score)."""
    weights = weights or ACT_WEIGHTS
    votes: Dict[str, float] = defaultdict(float)
    for window in per_window:
        best: Dict[str, float] = {}
        taken = 0
        for act, section in window:
            weight = weights.get(act, 0.0)
            if not weight:
                continue
            taken += 1
            if taken > depth:
                break
            for number in numbers_for(act, section):
                best[number] = max(best.get(number, 0.0), weight / (rrf_k + taken))
        for number, value in best.items():
            votes[number] += value
    return sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))


async def rank_sections_from_facts(
    sentences: Sequence[str], windows: int = WINDOWS
) -> List[Tuple[str, float]]:
    """Ranked (bare section number, score) for a narrative given as sentences."""
    from app.tools.legal_retrieval import retrieve_statutes

    per_window: List[List[Tuple[str, str]]] = []
    for window in fact_windows(sentences, window=4, stride=3, max_windows=windows):
        result, _parsed = await retrieve_statutes(
            window,
            k=RETRIEVAL_K,
            acts=list(ACT_WEIGHTS),
            min_score=0.0,
            candidate_pool=CANDIDATE_POOL,
            rerank_pool=RERANK_POOL,
        )
        per_window.append(
            [(c.act_name, c.section_number.replace("Article", "").strip().upper()) for c in result.chunks]
        )
    return aggregate(per_window)
