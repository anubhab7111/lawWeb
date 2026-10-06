"""Missing-detail checks for the find-lawyer and crime-report flows.

When a request lacks what the flow needs (a lawyer request with no place or no kind of
matter; a crime report too short to say what happened), the chatbot asks once, then
resumes the same flow with the user's reply merged into the original request.
"""

from functools import lru_cache
from typing import List, Optional

from app.text_match import any_word, contains_word

STATES_AND_UTS = [
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh", "goa", "gujarat", "haryana",
    "himachal pradesh", "jharkhand", "karnataka", "kerala", "madhya pradesh", "maharashtra", "manipur",
    "meghalaya", "mizoram", "nagaland", "odisha", "punjab", "rajasthan", "sikkim", "tamil nadu", "telangana",
    "tripura", "uttar pradesh", "uttarakhand", "west bengal", "andaman and nicobar", "chandigarh",
    "dadra and nagar haveli", "daman and diu", "delhi", "jammu and kashmir", "ladakh", "lakshadweep",
    "puducherry", "pondicherry",
]
# Common alternative names that people type but the lawyer table may not use.
PLACE_ALIASES = [
    "bombay", "mumbai", "calcutta", "kolkata", "madras", "chennai", "bangalore", "bengaluru", "gurgaon",
    "gurugram", "noida", "new delhi", "hyderabad", "pune", "ncr", "navi mumbai", "thane",
]
# A message naming any of these practice areas or matters says what the case is about.
MATTER_TERMS = [
    "criminal", "civil", "family", "divorce", "custody", "maintenance", "alimony", "dowry", "498a", "marriage",
    "adoption", "property", "land", "tenant", "landlord", "rent", "eviction", "builder", "rera", "flat",
    "partition", "will", "inheritance", "succession", "bail", "fir", "arrest", "cheque", "cheating", "fraud",
    "consumer", "tax", "gst", "income tax", "labour", "labor", "employment", "salary", "termination",
    "corporate", "company", "startup", "shareholder", "insolvency", "nclt", "contract", "agreement",
    "arbitration", "banking", "loan", "sarfaesi", "cyber", "defamation", "intellectual property", "trademark",
    "copyright", "patent", "medical negligence", "accident", "motor accident", "insurance", "injury",
    "immigration", "visa", "passport", "constitutional", "writ", "rti", "environmental", "pension",
    "service matter", "pocso", "ndps", "harassment", "domestic violence", "assault", "theft", "murder",
    "notice", "legal notice", "court case", "appeal", "high court", "supreme court",
]
CRIME_MIN_WORDS = 8


@lru_cache(maxsize=1)
def _places() -> List[str]:
    places = set(STATES_AND_UTS) | set(PLACE_ALIASES)
    try:
        from sqlmodel import Session, select

        from app.db.engine import get_engine
        from app.db.models import Lawyer

        with Session(get_engine()) as session:
            for loc in session.exec(select(Lawyer.location).distinct()).all():
                city = (loc or "").split(",")[0].strip().lower()
                if len(city) >= 4 and city.replace(" ", "").isalpha():
                    places.add(city)
    except Exception as e:
        print(f"[followup] lawyer locations unavailable ({e}) — using states and major cities only")
    return sorted(places, key=len, reverse=True)


def find_location(text: str) -> Optional[str]:
    for place in _places():
        if contains_word(text, place):
            return place
    return None


def has_matter(text: str) -> bool:
    return any_word(text, MATTER_TERMS)


def missing_lawyer_details(text: str) -> List[str]:
    missing = []
    if find_location(text) is None:
        missing.append("location")
    if not has_matter(text):
        missing.append("matter")
    return missing


def crime_report_too_thin(text: str) -> bool:
    return len(text.split()) < CRIME_MIN_WORDS
