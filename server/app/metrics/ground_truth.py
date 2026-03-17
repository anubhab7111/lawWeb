"""
Ground truth dataset for evaluating the legal chatbot RAG pipeline.

Each entry contains:
  - query              : The exact test prompt
  - relevant_ipc_sections : IPC/CrPC sections the RAG system *should* surface
                            (used for Hit Rate@k and MRR)
  - relevant_keywords  : Legal terms/acts/concepts that must appear in a
                         high-quality answer (used for lightweight Context Recall)
  - expected_acts      : Statutes / constitutional provisions that should be cited
  - reference_answer   : A concise gold-standard answer used by the LLM judge for
                         Context Recall and Faithfulness evaluation
  - domain             : Broad legal domain tag for filtering/reporting

NOTE: relevant_ipc_sections intentionally contains ONLY sections indexed in the
local IPC FAISS store.  Constitutional, contract, evidence, and family-law
queries have empty lists because the RAG corpus is IPC-only; those queries are
evaluated purely on generation quality.
"""

from typing import List, Optional, TypedDict


class GroundTruthEntry(TypedDict):
    query: str
    relevant_ipc_sections: List[str]  # e.g. ["302", "307"]
    relevant_keywords: List[str]  # must-appear terms
    expected_acts: List[str]  # statutes to cite
    reference_answer: str  # concise gold answer
    domain: str  # constitutional | criminal | contract | …


# ---------------------------------------------------------------------------
# 18 ground-truth entries — one per TEST_PROMPTS item in test_chatbot.py
# ---------------------------------------------------------------------------

GROUND_TRUTH: List[GroundTruthEntry] = [
    # ── 1 ───────────────────────────────────────────────────────────────────
    {
        "query": (
            "Can Parliament pass a law restricting social media speech citing "
            "\u201cpublic order\u201d? How would courts test its constitutionality under "
            "Article 19?"
        ),
        "relevant_ipc_sections": [],  # constitutional question; IPC RAG N/A
        "relevant_keywords": [
            "Article 19(1)(a)",
            "Article 19(2)",
            "reasonable restriction",
            "public order",
            "proportionality",
            "Shreya Singhal",
            "nexus",
            "overbroad",
        ],
        "expected_acts": [
            "Constitution of India",
            "Article 19",
            "Information Technology Act",
        ],
        "reference_answer": (
            "Parliament may restrict free speech on social media under Article 19(2) "
            "only if the restriction is (a) imposed by law, (b) falls within one of "
            "the eight enumerated grounds (including public order), and (c) is "
            "reasonable and proportionate.  Courts apply a three-part test: "
            "legitimate aim, rational nexus between the law and the aim, and "
            "proportionality (the restriction must not be overbroad). The Supreme "
            "Court's Shreya Singhal v. Union of India (2015) struck down Section 66A "
            "of the IT Act as unconstitutional because it was vague and overbroad, "
            "going far beyond what is necessary to protect public order."
        ),
        "domain": "constitutional",
    },
    # ── 2 ───────────────────────────────────────────────────────────────────
    {
        "query": (
            "Is the Right to Privacy absolute in India? Under what circumstances "
            "can the State legally conduct surveillance?"
        ),
        "relevant_ipc_sections": [],
        "relevant_keywords": [
            "Article 21",
            "K.S. Puttaswamy",
            "proportionality",
            "legitimate aim",
            "surveillance",
            "Telegraph Act",
            "not absolute",
            "procedural safeguard",
        ],
        "expected_acts": [
            "Constitution of India",
            "Article 21",
            "Indian Telegraph Act",
            "Information Technology Act",
        ],
        "reference_answer": (
            "The Right to Privacy is a fundamental right under Article 21 of the "
            "Constitution (K.S. Puttaswamy v. Union of India, 2017), but it is not "
            "absolute. State surveillance is permissible when it satisfies a "
            "three-pronged test: (1) legality — authorised by a valid law; "
            "(2) legitimate aim — national security, public order, prevention of "
            "crime, etc.; and (3) proportionality — the least intrusive means "
            "available. Lawful interception can be ordered under Section 5(2) of the "
            "Indian Telegraph Act or Section 69 of the IT Act with prior approval "
            "from the Home Secretary."
        ),
        "domain": "constitutional",
    },
    # ── 3 ───────────────────────────────────────────────────────────────────
    {
        "query": (
            "Can a State government refuse to implement a Central law? "
            "What remedies exist?"
        ),
        "relevant_ipc_sections": [],
        "relevant_keywords": [
            "Article 254",
            "Article 256",
            "repugnancy",
            "Concurrent List",
            "President's assent",
            "federal supremacy",
            "Article 365",
            "writ of mandamus",
        ],
        "expected_acts": [
            "Constitution of India",
            "Article 254",
            "Article 256",
            "Article 365",
        ],
        "reference_answer": (
            "A State government cannot ordinarily refuse to implement a validly "
            "enacted Central law. Under Article 256, every State is obligated to "
            "exercise its executive power in compliance with Central laws. Under "
            "Article 254, if a State law on the Concurrent List conflicts with a "
            "Central law, the Central law prevails unless the State law received "
            "Presidential assent. Remedies include: (1) the Union directing the "
            "State under Article 256; (2) failure exposing the State to action "
            "under Article 365 (failure of constitutional machinery); and "
            "(3) an aggrieved party filing a writ of mandamus before the High "
            "Court or Supreme Court to compel compliance."
        ),
        "domain": "constitutional",
    },
    # ── 4 ───────────────────────────────────────────────────────────────────
    {
        "query": "How does the \u201cbasic structure doctrine\u201d limit constitutional amendments?",
        "relevant_ipc_sections": [],
        "relevant_keywords": [
            "Kesavananda Bharati",
            "Article 368",
            "basic structure",
            "implied limitation",
            "judicial review",
            "separation of powers",
            "democratic republic",
            "void",
        ],
        "expected_acts": [
            "Constitution of India",
            "Article 368",
        ],
        "reference_answer": (
            "The basic structure doctrine, established in Kesavananda Bharati v. State "
            "of Kerala (1973), holds that while Parliament has wide amending power under "
            "Article 368, it cannot alter or destroy the 'basic structure' of the "
            "Constitution. Features forming the basic structure include: supremacy of "
            "the Constitution, republican and democratic form of government, secular "
            "character, separation of powers, and judicial review. Any constitutional "
            "amendment that abrogates these features is void and will be struck down "
            "by the Supreme Court. The doctrine acts as an implied limitation on "
            "Parliament's amending power."
        ),
        "domain": "constitutional",
    },
    # ── 5 ───────────────────────────────────────────────────────────────────
    {
        "query": "Can an FIR be quashed by the High Court? On what grounds?",
        "relevant_ipc_sections": [],
        "relevant_keywords": [
            "Section 482 CrPC",
            "inherent powers",
            "Bhajan Lal",
            "no cognizable offence",
            "abuse of process",
            "settlement",
            "ex facie false",
            "quash",
        ],
        "expected_acts": [
            "Code of Criminal Procedure 1973",
            "Section 482 CrPC",
        ],
        "reference_answer": (
            "Yes. A High Court may quash an FIR under Section 482 CrPC (inherent "
            "powers) or under Article 226 of the Constitution. The Supreme Court in "
            "State of Haryana v. Bhajan Lal (1992) laid down exhaustive guidelines, "
            "including: (1) the allegations, even taken at face value, do not "
            "constitute any offence; (2) the FIR is ex facie false or filed with a "
            "mala fide motive; (3) the offence is non-cognizable and no Magistrate's "
            "permission was obtained; (4) the dispute is purely civil in nature; "
            "(5) the parties have settled and continuation of proceedings would be "
            "an abuse of process. Courts must exercise this power sparingly and with "
            "caution."
        ),
        "domain": "criminal_procedure",
    },
    # ── 6 ───────────────────────────────────────────────────────────────────
    {
        "query": "Is anticipatory bail available for economic offences?",
        "relevant_ipc_sections": ["420", "406", "409", "467", "468", "471"],
        "relevant_keywords": [
            "Section 438 CrPC",
            "anticipatory bail",
            "economic offence",
            "gravity",
            "class apart",
            "flight risk",
            "money laundering",
            "special court",
            "prima facie",
        ],
        "expected_acts": [
            "Code of Criminal Procedure 1973",
            "Section 438 CrPC",
            "Prevention of Money Laundering Act",
            "Indian Penal Code",
        ],
        "reference_answer": (
            "Anticipatory bail under Section 438 CrPC is not automatically barred for "
            "economic offences, but courts apply heightened scrutiny. The Supreme Court "
            "has consistently held that economic offences 'constitute a class apart' "
            "and 'have serious repercussions on the financial health of the community'. "
            "Relevant factors include: gravity of the alleged fraud, amount involved, "
            "flight risk, likelihood of tampering with evidence, and whether special "
            "statutes (like PMLA) carry explicit bars on anticipatory bail. Under "
            "PMLA, for instance, the twin conditions in Section 45 make bail "
            "extremely restrictive. For pure IPC economic offences (e.g., Section 420 "
            "or 406), anticipatory bail is possible but courts grant it cautiously."
        ),
        "domain": "criminal",
    },
    # ── 7 ───────────────────────────────────────────────────────────────────
    {
        "query": "Can a criminal case proceed if the complainant withdraws?",
        "relevant_ipc_sections": [],
        "relevant_keywords": [
            "Section 321 CrPC",
            "public prosecutor",
            "State vs accused",
            "compoundable",
            "non-compoundable",
            "Section 320 CrPC",
            "magistrate discretion",
            "consent",
        ],
        "expected_acts": [
            "Code of Criminal Procedure 1973",
            "Section 320 CrPC",
            "Section 321 CrPC",
        ],
        "reference_answer": (
            "In most criminal cases, the State is the complainant and the victim is "
            "only a witness; therefore a victim's withdrawal does not automatically "
            "end the case. Under Section 321 CrPC, only the Public Prosecutor (with "
            "the court's consent) can withdraw a case. The Magistrate may permit "
            "compounding of compoundable offences (listed in Section 320 CrPC) on "
            "the parties' agreement, but non-compoundable offences — such as murder "
            "or rape — cannot be settled privately and the case proceeds regardless "
            "of the complainant's wishes. Even for compoundable offences, the court "
            "must be satisfied that the composition is genuine and not coerced."
        ),
        "domain": "criminal_procedure",
    },
    # ── 8 ───────────────────────────────────────────────────────────────────
    {
        "query": (
            "Does marital rape constitute an offence in India? "
            "Explain the legal position."
        ),
        "relevant_ipc_sections": ["375", "376", "376B"],
        "relevant_keywords": [
            "Section 375 IPC",
            "Exception 2",
            "marital exemption",
            "RIT Foundation",
            "domestic violence",
            "Protection of Women",
            "unconstitutional",
            "contested",
        ],
        "expected_acts": [
            "Indian Penal Code",
            "Section 375 IPC",
            "Protection of Women from Domestic Violence Act 2005",
        ],
        "reference_answer": (
            "Under current Indian law, marital rape of a wife above 18 years is not "
            "a criminal offence under the IPC due to Exception 2 to Section 375, "
            "which exempts a husband from rape charges when the wife is above 18. "
            "However, a wife who is judicially separated can file a complaint under "
            "Section 376B IPC (sexual intercourse by husband upon his wife during "
            "separation). Non-consensual sex within marriage may be addressed under "
            "the Protection of Women from Domestic Violence Act 2005 (civil remedy) "
            "or Section 498A IPC (cruelty). The Delhi High Court delivered a split "
            "verdict in RIT Foundation v. Union of India (2022), and the matter is "
            "pending before the Supreme Court. The legal position is therefore "
            "contested and may change."
        ),
        "domain": "criminal",
    },
    # ── 9 ───────────────────────────────────────────────────────────────────
    {
        "query": (
            "Can cryptocurrency transactions attract criminal liability "
            "under existing Indian laws?"
        ),
        "relevant_ipc_sections": ["420", "406", "465", "468", "471", "120B"],
        "relevant_keywords": [
            "IT Act",
            "Section 66",
            "PMLA",
            "FEMA",
            "cheating",
            "Section 420 IPC",
            "criminal breach of trust",
            "unregulated",
            "virtual digital asset",
        ],
        "expected_acts": [
            "Indian Penal Code",
            "Information Technology Act 2000",
            "Prevention of Money Laundering Act",
            "Foreign Exchange Management Act",
        ],
        "reference_answer": (
            "Cryptocurrency transactions can attract criminal liability under several "
            "existing Indian laws even in the absence of a dedicated crypto statute. "
            "If the transaction involves cheating or deceit, Section 420 IPC applies. "
            "If it involves criminal breach of trust (e.g., a crypto exchange "
            "misappropriating customer funds), Section 406/409 IPC applies. "
            "Money-laundering through crypto is covered by PMLA 2002. Cross-border "
            "crypto transfers may attract FEMA. Section 66C/66D of the IT Act covers "
            "identity theft and phishing in crypto frauds. Importantly, the Supreme "
            "Court in Internet and Mobile Association of India v. RBI (2020) struck "
            "down RBI's banking ban on crypto, meaning crypto itself is not illegal, "
            "but fraudulent use remains criminal."
        ),
        "domain": "technology_law",
    },
    # ── 10 ──────────────────────────────────────────────────────────────────
    {
        "query": (
            "If someone's private photos are shared online without consent, "
            "what legal remedies are available?"
        ),
        "relevant_ipc_sections": ["354C", "499", "509", "509"],
        "relevant_keywords": [
            "Section 66E IT Act",
            "Section 67 IT Act",
            "Section 354C IPC",
            "revenge porn",
            "voyeurism",
            "FIR",
            "cyber crime cell",
            "non-consensual",
            "Section 67A IT Act",
        ],
        "expected_acts": [
            "Information Technology Act 2000",
            "Indian Penal Code",
            "Section 66E IT Act",
            "Section 354C IPC",
        ],
        "reference_answer": (
            "Multiple legal remedies exist. (1) Section 66E of the IT Act (violation "
            "of privacy by capturing/publishing private images without consent) — "
            "imprisonment up to 3 years and fine. (2) Section 67/67A IT Act for "
            "publishing obscene/sexually explicit material online. (3) Section 354C "
            "IPC (voyeurism) — imprisonment 1-3 years (first offence), 3-7 years "
            "(repeat). (4) Section 509 IPC (words/gesture intended to insult a "
            "woman's modesty). The victim should: file an FIR at the local police "
            "station or the Cyber Crime Cell; lodge a complaint on "
            "cybercrime.gov.in; and approach the platform to take down the content. "
            "Courts can also grant injunctive relief and compensation in civil "
            "proceedings for privacy violations."
        ),
        "domain": "technology_law",
    },
    # ── 11 ──────────────────────────────────────────────────────────────────
    {
        "query": (
            "Who is liable if an AI system causes financial loss — "
            "developer, deployer, or user?"
        ),
        "relevant_ipc_sections": ["420", "304A"],
        "relevant_keywords": [
            "product liability",
            "Consumer Protection Act 2019",
            "negligence",
            "Section 2(34) CPA",
            "vicarious liability",
            "developer",
            "deployer",
            "contractual indemnity",
            "no specific AI law",
        ],
        "expected_acts": [
            "Consumer Protection Act 2019",
            "Indian Contract Act 1872",
            "Indian Penal Code",
            "Information Technology Act 2000",
        ],
        "reference_answer": (
            "India currently has no dedicated AI liability statute, so liability is "
            "determined under existing frameworks. (1) Developer liability: Under the "
            "Consumer Protection Act 2019, Section 2(34), a 'product' includes "
            "software; if an AI product has a design defect, the manufacturer/developer "
            "can be held strictly liable. Negligence principles under tort law also "
            "apply. (2) Deployer liability: The entity deploying the AI system in a "
            "commercial context owes a duty of care to users; failure to supervise or "
            "set proper guardrails can constitute negligence. (3) User liability: If a "
            "user provides misleading inputs that cause the AI to produce harmful "
            "outputs, they may bear partial liability. In practice, liability is often "
            "shared or allocated by contract. Where the AI commits cheating or fraud, "
            "Section 420 IPC may be invoked against the controlling person."
        ),
        "domain": "technology_law",
    },
    # ── 12 ──────────────────────────────────────────────────────────────────
    {
        "query": "Are WhatsApp chats admissible as evidence in Indian courts?",
        "relevant_ipc_sections": [],
        "relevant_keywords": [
            "Section 65B Indian Evidence Act",
            "electronic record",
            "certificate",
            "primary evidence",
            "authenticity",
            "Arjun Panditrao",
            "Anvar PV",
            "hash value",
        ],
        "expected_acts": [
            "Indian Evidence Act 1872",
            "Section 65B IEA",
            "Information Technology Act 2000",
        ],
        "reference_answer": (
            "WhatsApp chats are electronic records and can be admitted as evidence "
            "under Section 65B of the Indian Evidence Act 1872. For admissibility "
            "the party seeking to rely on them must produce a certificate under "
            "Section 65B(4) attesting that the electronic record was produced by a "
            "computer in the ordinary course of activities and the computer was "
            "operating properly. The Supreme Court in Arjun Panditrao Khotkar v. "
            "Kailash Kushanrao Gorantyal (2020) confirmed this certificate is "
            "mandatory when chats are produced as secondary evidence. Authentication "
            "issues — such as proving who actually sent the messages — must also be "
            "established, typically through forensic analysis or admission."
        ),
        "domain": "evidence_law",
    },
    # ── 13 ──────────────────────────────────────────────────────────────────
    {
        "query": (
            "Is a contract enforceable if signed under economic pressure "
            "but without explicit coercion?"
        ),
        "relevant_ipc_sections": [],
        "relevant_keywords": [
            "Section 16 ICA",
            "undue influence",
            "Section 19A ICA",
            "free consent",
            "economic duress",
            "voidable",
            "dominant position",
            "unconscionable bargain",
        ],
        "expected_acts": [
            "Indian Contract Act 1872",
            "Section 16 ICA",
            "Section 19A ICA",
        ],
        "reference_answer": (
            "A contract is voidable (not void) if consent was obtained by undue "
            "influence under Section 16 of the Indian Contract Act 1872. Undue "
            "influence occurs when one party is in a position to dominate the will "
            "of the other and uses that position to obtain an unfair advantage. "
            "'Economic pressure' alone — such as a desperate financial situation — "
            "does not automatically vitiate consent; courts look for a 'dominant "
            "position' relationship between the parties. If the weaker party can "
            "prove that the contract was unconscionable and the other party "
            "exploited the situation, the contract is voidable under Section 19A "
            "and may be set aside. Pure economic hardship without a dominating "
            "relationship generally does not suffice."
        ),
        "domain": "contract_law",
    },
    # ── 14 ──────────────────────────────────────────────────────────────────
    {
        "query": "Can an oral agreement be legally binding in India?",
        "relevant_ipc_sections": [],
        "relevant_keywords": [
            "Section 10 ICA",
            "offer and acceptance",
            "consideration",
            "free consent",
            "written requirement",
            "Registration Act",
            "immovable property",
            "enforceable",
        ],
        "expected_acts": [
            "Indian Contract Act 1872",
            "Section 10 ICA",
            "Transfer of Property Act 1882",
            "Registration Act 1908",
        ],
        "reference_answer": (
            "Yes, an oral agreement is legally binding in India provided it satisfies "
            "the essentials of a valid contract under Section 10 of the Indian Contract "
            "Act 1872: offer and acceptance, lawful consideration, free consent of "
            "competent parties, and a lawful object. The Act does not generally require "
            "contracts to be in writing. However, important exceptions exist: contracts "
            "relating to immovable property above INR 100 must be in writing and "
            "registered under the Registration Act 1908 and the Transfer of Property "
            "Act; negotiable instruments require writing; and arbitration agreements "
            "must be in writing under the Arbitration and Conciliation Act 1996. "
            "Proving an oral contract in court relies on witness testimony and "
            "corroborating conduct."
        ),
        "domain": "contract_law",
    },
    # ── 15 ──────────────────────────────────────────────────────────────────
    {
        "query": "What happens if one party breaches a contract but claims \u201cforce majeure\u201d?",
        "relevant_ipc_sections": [],
        "relevant_keywords": [
            "Section 56 ICA",
            "frustration",
            "force majeure clause",
            "supervening impossibility",
            "Satyabrata Ghose",
            "beyond control",
            "void",
            "discharge of contract",
            "foreseeability",
        ],
        "expected_acts": [
            "Indian Contract Act 1872",
            "Section 56 ICA",
        ],
        "reference_answer": (
            "If a force majeure clause exists in the contract, its scope and "
            "consequences are governed by the clause itself; the courts will enforce "
            "it as written. If no such clause exists, the doctrine of frustration "
            "under Section 56 of the Indian Contract Act 1872 may apply. Under "
            "Section 56, a contract becomes void when performance becomes impossible "
            "or unlawful due to a supervening event that was unforeseeable and beyond "
            "the parties' control (Satyabrata Ghose v. Mugneeram Bangur, 1954). "
            "Mere difficulty or economic hardship does not amount to frustration. "
            "If force majeure is validly invoked, the contract is discharged and "
            "neither party is liable for damages; if not valid, the breaching party "
            "remains liable for damages under Section 73 ICA."
        ),
        "domain": "contract_law",
    },
    # ── 16 ──────────────────────────────────────────────────────────────────
    {
        "query": "Is a non-compete clause valid after employment ends?",
        "relevant_ipc_sections": [],
        "relevant_keywords": [
            "Section 27 ICA",
            "restraint of trade",
            "void",
            "post-employment",
            "reasonable",
            "Niranjan Shankar Golikari",
            "trade secret",
            "limited duration",
        ],
        "expected_acts": [
            "Indian Contract Act 1872",
            "Section 27 ICA",
        ],
        "reference_answer": (
            "Post-employment non-compete clauses are generally void in India under "
            "Section 27 of the Indian Contract Act 1872, which declares void any "
            "agreement that restrains a person from exercising a lawful profession, "
            "trade, or business. Unlike in many common law jurisdictions, Indian "
            "courts do not apply a 'reasonableness' test to post-employment "
            "restraints — they are void per se (Niranjan Shankar Golikari v. Century "
            "Spinning, 1967). However, the Supreme Court distinguished between "
            "restrictions during the employment period (valid if reasonable) and "
            "post-employment restrictions (void). Narrow clauses protecting trade "
            "secrets or confidential information may be enforceable, but a blanket "
            "bar on working in the same industry after resignation is not."
        ),
        "domain": "contract_law",
    },
    # ── 17 ──────────────────────────────────────────────────────────────────
    {
        "query": "Can ancestral property be sold without consent of all legal heirs?",
        "relevant_ipc_sections": [],
        "relevant_keywords": [
            "Hindu Succession Act",
            "Section 6 HSA",
            "coparcener",
            "Mitakshara",
            "undivided share",
            "partition",
            "alienation",
            "karta",
            "voidable",
        ],
        "expected_acts": [
            "Hindu Succession Act 1956",
            "Transfer of Property Act 1882",
            "Hindu Succession (Amendment) Act 2005",
        ],
        "reference_answer": (
            "Under Hindu personal law, ancestral (coparcenary) property cannot be "
            "validly sold by one coparcener alone without the consent of all "
            "coparceners. Section 6 of the Hindu Succession Act 1956, as amended in "
            "2005, grants daughters equal coparcenary rights. A karta (manager of a "
            "Hindu Undivided Family) can alienate ancestral property only for legal "
            "necessity or for the benefit of the estate. Any other alienation without "
            "consent of all coparceners is voidable at the option of the dissenting "
            "coparceners. A co-sharer can file a suit for partition and separate "
            "possession of their undivided share, or challenge the sale deed in court."
        ),
        "domain": "property_law",
    },
    # ── 18 ──────────────────────────────────────────────────────────────────
    {
        "query": "What legal rights does a live-in partner have over shared property?",
        "relevant_ipc_sections": [],
        "relevant_keywords": [
            "Protection of Women from Domestic Violence Act 2005",
            "Section 2(f) PWDVA",
            "domestic relationship",
            "Section 20 PWDVA",
            "maintenance",
            "shared household",
            "D Velusamy",
            "legitimate expectation",
        ],
        "expected_acts": [
            "Protection of Women from Domestic Violence Act 2005",
            "Indian Succession Act 1925",
            "Section 125 CrPC",
        ],
        "reference_answer": (
            "Live-in partners do not have the same automatic property rights as "
            "married spouses. However, several protections exist. Under the "
            "Protection of Women from Domestic Violence Act 2005, a female partner "
            "in a 'relationship in the nature of marriage' (Section 2(f) PWDVA) can "
            "claim the right to reside in the shared household, maintenance, and "
            "monetary relief (Section 20). The Supreme Court in D. Velusamy v. D. "
            "Patchaiammal (2010) held that to qualify as a 'relationship in the "
            "nature of marriage' the couple must have lived together for a "
            "significant period and presented themselves as spouses. Property "
            "contributed jointly (both names on title, or provable financial "
            "contribution) can be claimed through a civil suit for partition or "
            "declaration. A live-in partner has no inheritance rights under Hindu "
            "personal law but may be a beneficiary under a Will."
        ),
        "domain": "family_law",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_entry_by_query(query: str) -> Optional[GroundTruthEntry]:
    """Return the ground-truth entry whose query matches (exact or stripped)."""
    q = query.strip()
    for entry in GROUND_TRUTH:
        if entry["query"].strip() == q:
            return entry
    return None


def get_entries_by_domain(domain: str) -> List[GroundTruthEntry]:
    """Return all entries belonging to a given domain."""
    return [e for e in GROUND_TRUTH if e["domain"] == domain]


def get_all_domains() -> List[str]:
    """Return a sorted, deduplicated list of all domains in the dataset."""
    return sorted({e["domain"] for e in GROUND_TRUTH})


# Quick self-check
if __name__ == "__main__":
    print(f"Ground truth entries: {len(GROUND_TRUTH)}")
    for i, entry in enumerate(GROUND_TRUTH, 1):
        secs = entry["relevant_ipc_sections"]
        print(
            f"  [{i:02d}] domain={entry['domain']:<20} "
            f"ipc_sections={secs if secs else '[]':<20} "
            f"query={entry['query'][:60]}..."
        )
