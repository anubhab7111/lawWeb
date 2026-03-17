"""
Crime reporting module.
Provides crime type detection for routing crime reports.
The finetuned LLM handles generating guidance, IPC sections, punishment, and further steps.
"""

from typing import Dict, List


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
        score = sum(1 for keyword in keywords if keyword in description_lower)
        if score > 0:
            scores[crime_type] = score

    if scores:
        return max(scores.items(), key=lambda x: x[1])[0]

    return "general"


def is_complex_crime(crime_type: str) -> bool:
    """Check if a crime type is complex enough to warrant RAG lookup for IPC sections."""
    return crime_type in COMPLEX_CRIMES
