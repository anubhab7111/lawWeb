"""
Crime reporting module.
Provides crime type detection for routing crime reports.
The finetuned LLM handles generating guidance, IPC sections, punishment, and further steps.
"""

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.text_match import count_words

CLASSIFIER_DIR = Path(__file__).resolve().parents[1] / "data" / "crime_type_classifier"

# List of recognized crime types for the /crime-types API endpoint
CRIME_TYPES: List[str] = [
    "theft",
    "robbery",
    "assault",
    "fraud",
    "cheating",
    "harassment",
    "threat",
    "cybercrime",
    "domestic_violence",
    "property_damage",
    "land_dispute",
    "arson",
    "murder",
    "kidnapping",
    "rape",
    "dowry",
]

# Keywords for crime type detection
CRIME_KEYWORDS: Dict[str, List[str]] = {
    "threat": [
        "threatened",
        "threatening",
        "threat",
        "threats",
        "intimidation",
        "intimidate",
        "intimidated",
        "warned me",
        "will kill",
        "threatened to kill",
        "death threat",
        "life threat",
    ],
    "theft": [
        "stolen",
        "theft",
        "robbery",
        "burglary",
        "shoplifting",
        "pickpocket",
        "mugged",
        "break-in",
        "stole",
        "bike stolen",
        "car stolen",
        "phone stolen",
    ],
    "robbery": [
        "robbery",
        "robbed",
        "forcefully took",
        "mugged",
        "loot",
    ],
    "assault": [
        "assault",
        "attacked",
        "beaten",
        "hit",
        "punched",
        "physical violence",
        "injured",
        "hurt by someone",
    ],
    "fraud": [
        "scam",
        "fraud",
        "scammed",
        "swindled",
        "phishing",
        "fake",
        "deceived",
        "money stolen",
        "cheating",
        "cheated",
    ],
    "harassment": [
        "harassing",
        "harassment",
        "stalking",
        "bullying",
        "eve teasing",
    ],
    "cybercrime": [
        "hacked",
        "hacking",
        "malware",
        "ransomware",
        "identity theft",
        "phishing",
        "online scam",
        "data breach",
        "cyber",
    ],
    "domestic_violence": [
        "domestic violence",
        "spouse abuse",
        "partner abuse",
        "family violence",
        "abusive relationship",
        "abusive husband",
        "abusive wife",
    ],
    "property_damage": [
        "vandalism",
        "property damage",
        "damaged my",
        "graffiti",
        "broken window",
        "keyed car",
    ],
    "land_dispute": [
        "land",
        "property grabbed",
        "encroachment",
        "land grabbing",
        "trespass",
        "illegally taken",
        "land taken",
        "property taken",
        "occupied my land",
        "encroached",
    ],
    "arson": [
        "fire",
        "arson",
        "set fire",
        "burning",
        "burnt",
        "on fire",
        "flames",
        "house fire",
    ],
    "murder": [
        "murder",
        "killed",
        "homicide",
        "dead body",
        "dead",
        "died",
    ],
    "kidnapping": [
        "kidnapping",
        "kidnapped",
        "abducted",
        "abduction",
        "ransom",
        "missing person",
    ],
    "rape": [
        "rape",
        "raped",
        "sexual assault",
        "molestation",
        "molested",
        "sexually assaulted",
    ],
    "dowry": [
        "dowry",
        "dowry harassment",
        "dowry death",
        "cruelty by husband",
        "in-laws harassing",
        "498a",
    ],
}

# Complex crimes that benefit from RAG lookup for IPC sections
COMPLEX_CRIMES: List[str] = [
    "land_dispute",
    "cybercrime",
    "domestic_violence",
    "dowry",
    "fraud",
]


def detect_crime_type(description: str) -> str:
    """
    Detect the type of crime based on description keywords.

    Args:
        description: User's description of the incident

    Returns:
        Detected crime type string
    """
    description_lower = description.lower()

    scores: Dict[str, int] = {}
    for crime_type, keywords in CRIME_KEYWORDS.items():
        score = count_words(description_lower, keywords)
        if score > 0:
            scores[crime_type] = score

    if scores:
        return max(scores.items(), key=lambda x: x[1])[0]

    return "general"


# Classifier labels are the IndianBailJudgments-1200 crime families it was trained on.
# Two families cover more than one of our types; keywords pick within those (first wins ties).
FAMILY_TYPES: Dict[str, List[str]] = {
    "Theft or Robbery": ["theft", "robbery"],
    "Dowry Harassment": ["dowry"],
    "Sexual Offense": ["harassment", "rape"],
    "Fraud or Cheating": ["fraud"],
    "Cyber Crime": ["cybercrime"],
    "Extortion": ["threat"],
    "Kidnapping": ["kidnapping"],
    "Murder": ["murder"],
    "Domestic Violence": ["domestic_violence"],
    "Narcotics": ["general"],
    "Others": ["general"],
}


@lru_cache(maxsize=1)
def _load_classifier() -> Optional[Tuple[np.ndarray, np.ndarray, List[str]]]:
    weights = CLASSIFIER_DIR / "weights.npz"
    if not weights.exists():
        return None
    z = np.load(weights)
    return z["W"], z["b"], [str(label) for label in z["labels"]]


ANIMAL_WORDS = [
    "dog", "dogs", "puppy", "cat", "cats", "kitten", "cow", "cows", "buffalo", "goat", "goats",
    "horse", "ox", "bull", "cattle", "pet", "animal", "animals", "hen", "hens", "sheep",
]
HUMAN_VICTIM_WORDS = [
    "son", "daughter", "brother", "sister", "father", "mother", "husband", "wife", "child",
    "baby", "uncle", "aunt", "cousin", "friend", "grandfather", "grandmother", "man", "woman",
    "boy", "girl", "person", "people", "victim", "victims", "student", "body",
]


def resolve_family(family: str, description: str) -> str:
    text = description.lower()
    if family == "Murder":
        # Bail data files threats under Extortion, so "threatening to kill me" scores as
        # Murder; with no sign that anyone died it is criminal intimidation.
        died = count_words(text, CRIME_KEYWORDS["murder"])
        if not died and count_words(text, CRIME_KEYWORDS["threat"]):
            return "threat"
        # Killing someone's animal is mischief (IPC 428/429, BNS 325), not murder.
        if count_words(text, ANIMAL_WORDS) and not count_words(text, HUMAN_VICTIM_WORDS):
            return "property_damage"
        return "murder"
    return max(FAMILY_TYPES[family], key=lambda t: count_words(text, CRIME_KEYWORDS.get(t, [])))


def crime_type_from_vector(q: np.ndarray, description: str) -> str:
    W, b, labels = _load_classifier()  # type: ignore[misc]
    return resolve_family(labels[int(np.argmax(W @ (q / np.linalg.norm(q)) + b))], description)


async def classify_crime_type(description: str) -> str:
    """Crime type from a logistic-regression head over the shared BGE-M3 embedding;
    falls back to keyword matching when the weights or the embedding model are missing."""
    if _load_classifier() is None:
        return detect_crime_type(description)
    try:
        from app.tools.base_legal_rag import _get_shared_embeddings

        embeddings = await _get_shared_embeddings()
        q = np.array(embeddings.embed_query(description), dtype=np.float32)
    except Exception as e:
        print(f"[crime_reporter] embedding failed ({e}) — keyword fallback")
        return detect_crime_type(description)
    return crime_type_from_vector(q, description)


def is_complex_crime(crime_type: str) -> bool:
    """Check if a crime type is complex enough to warrant RAG lookup for IPC sections."""
    return crime_type in COMPLEX_CRIMES
