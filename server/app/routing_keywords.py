"""
Keyword bank for the chatbot's multi-offense detection.

CRIME_TYPE_KEYWORDS is the sole survivor here — everything else that used
to live in this module (non-legal/legal domain gates, validation intent,
per-domain keyword banks feeding domain_hint) has been replaced by
embedding-based classifiers in app.intent_classifier. This one stays
keyword-based deliberately: it's a literal-match count feeding a retrieval
k-parameter (app.chatbot.handle_crime_report), not a classification
decision — counting how many distinct offense-type words appear in the
text isn't something embedding similarity is suited to.
"""

# Crime type keywords for multi-offense detection
CRIME_TYPE_KEYWORDS = frozenset(
    [
        "forgery",
        "forged",
        "trespass",
        "trespassed",
        "assault",
        "assaulted",
        "threat",
        "threatened",
        "bribe",
        "bribery",
        "fraud",
        "cyber",
        "identity theft",
        "launder",
        "laundering",
        "cheating",
        "theft",
        "robbery",
        "murder",
        "kidnapping",
        "extortion",
        "blackmail",
        "defamation",
        "harassment",
        "stalking",
        "dowry",
        "domestic violence",
    ]
)
