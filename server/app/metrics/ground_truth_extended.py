"""
Extended ground-truth entries covering the domains added by the unified
all-domain RAG index (family, labour, corporate, tax, environment, consumer,
cyber/IP, election, property, commercial, human rights).

Each entry carries `relevant_sections` — act-agnostic section/article numbers
used by the generalized Hit Rate@k / MRR scoring in evaluator.py.
"""
EXTENDED_GROUND_TRUTH = [
    # ── family ──────────────────────────────────────────────────────────────
    {
        "query": "What are the grounds for divorce under the Hindu Marriage Act?",
        "relevant_ipc_sections": [],
        "relevant_sections": ["13"],
        "relevant_keywords": [
            "Section 13",
            "Hindu Marriage Act",
            "cruelty",
            "desertion",
            "adultery",
            "conversion",
            "mental disorder",
            "mutual consent",
        ],
        "expected_acts": ["Hindu Marriage Act 1955"],
        "reference_answer": (
            "Section 13 of the Hindu Marriage Act 1955 lists the grounds for "
            "divorce: adultery, cruelty, desertion for two years, conversion to "
            "another religion, incurable mental disorder, virulent and incurable "
            "leprosy (removed in 2019), venereal disease, renunciation of the "
            "world, and not being heard alive for seven years. Section 13(2) adds "
            "wife-specific grounds (e.g., husband's bigamy, rape/sodomy/"
            "bestiality). Section 13B provides divorce by mutual consent with a "
            "six-month cooling-off period that courts may waive."
        ),
        "domain": "family_law",
    },
    {
        "query": "Can elderly parents claim maintenance from their children in India?",
        "relevant_ipc_sections": [],
        "relevant_sections": ["4", "5", "125"],
        "relevant_keywords": [
            "Maintenance and Welfare of Parents and Senior Citizens Act 2007",
            "Section 4",
            "maintenance tribunal",
            "Section 125 CrPC",
            "monthly allowance",
            "children",
            "senior citizen",
        ],
        "expected_acts": [
            "Maintenance and Welfare of Parents and Senior Citizens Act 2007",
            "Code of Criminal Procedure 1973",
        ],
        "reference_answer": (
            "Yes. Under Section 4 of the Maintenance and Welfare of Parents and "
            "Senior Citizens Act 2007, parents and senior citizens unable to "
            "maintain themselves can claim maintenance from their children (and "
            "childless senior citizens from relatives who would inherit). "
            "Applications go to a Maintenance Tribunal under Section 5, which can "
            "order a monthly allowance and decide within 90 days. Section 125 "
            "CrPC independently allows parents to claim maintenance from children "
            "with sufficient means."
        ),
        "domain": "family_law",
    },
    # ── labour ──────────────────────────────────────────────────────────────
    {
        "query": "How much maternity leave is a woman employee entitled to in India?",
        "relevant_ipc_sections": [],
        "relevant_sections": ["5"],
        "relevant_keywords": [
            "Maternity Benefit Act",
            "Section 5",
            "26 weeks",
            "12 weeks",
            "two surviving children",
            "adopting mother",
            "creche",
        ],
        "expected_acts": ["Maternity Benefit Act 1961"],
        "reference_answer": (
            "Under Section 5 of the Maternity Benefit Act 1961 (as amended in "
            "2017), a woman employee is entitled to 26 weeks of paid maternity "
            "leave for the first two children, of which up to 8 weeks may be "
            "taken before delivery. For a third or subsequent child the "
            "entitlement is 12 weeks. An adopting mother (child below 3 months) "
            "and a commissioning mother get 12 weeks. Establishments with 50+ "
            "employees must provide creche facilities."
        ),
        "domain": "labour_law",
    },
    {
        "query": "What compensation must an employer pay when retrenching a workman?",
        "relevant_ipc_sections": [],
        "relevant_sections": ["25F"],
        "relevant_keywords": [
            "Industrial Disputes Act",
            "Section 25F",
            "retrenchment",
            "one month notice",
            "fifteen days average pay",
            "continuous service",
            "compensation",
        ],
        "expected_acts": ["Industrial Disputes Act 1947"],
        "reference_answer": (
            "Section 25F of the Industrial Disputes Act 1947 makes retrenchment "
            "of a workman with at least one year of continuous service invalid "
            "unless: (a) one month's written notice stating reasons (or pay in "
            "lieu) is given; (b) retrenchment compensation equal to 15 days' "
            "average pay for every completed year of continuous service is paid "
            "at the time of retrenchment; and (c) notice is served on the "
            "appropriate Government. Non-compliance renders the retrenchment "
            "void ab initio and the workman is entitled to reinstatement."
        ),
        "domain": "labour_law",
    },
    # ── corporate ───────────────────────────────────────────────────────────
    {
        "query": "How can a financial creditor initiate corporate insolvency proceedings against a defaulting company?",
        "relevant_ipc_sections": [],
        "relevant_sections": ["7"],
        "relevant_keywords": [
            "Insolvency and Bankruptcy Code",
            "Section 7",
            "financial creditor",
            "NCLT",
            "default",
            "one crore",
            "CIRP",
            "moratorium",
        ],
        "expected_acts": ["Insolvency and Bankruptcy Code 2016"],
        "reference_answer": (
            "A financial creditor can file an application under Section 7 of the "
            "Insolvency and Bankruptcy Code 2016 before the National Company Law "
            "Tribunal (NCLT) when a corporate debtor defaults on a debt of at "
            "least INR 1 crore. The application must establish the default with "
            "records from an information utility or other evidence. On admission, "
            "the Corporate Insolvency Resolution Process (CIRP) begins, a "
            "moratorium under Section 14 takes effect, and an interim resolution "
            "professional takes over management."
        ),
        "domain": "corporate_law",
    },
    {
        "query": "What is the requirement for independent directors on the board of a listed company?",
        "relevant_ipc_sections": [],
        "relevant_sections": ["149"],
        "relevant_keywords": [
            "Companies Act 2013",
            "Section 149",
            "independent director",
            "one-third",
            "listed company",
            "board composition",
            "woman director",
        ],
        "expected_acts": ["Companies Act 2013"],
        "reference_answer": (
            "Section 149 of the Companies Act 2013 requires every listed public "
            "company to have at least one-third of its board as independent "
            "directors. Section 149(6) defines independence (no material "
            "pecuniary relationship, not a promoter, etc.). The section also "
            "mandates at least one woman director for prescribed companies, a "
            "resident director, and limits: minimum 3 directors (public), "
            "maximum 15 (extendable by special resolution). Independent "
            "directors serve up to two consecutive five-year terms."
        ),
        "domain": "corporate_law",
    },
    # ── indirect_tax ────────────────────────────────────────────────────────
    {
        "query": "What are the conditions for claiming input tax credit under GST?",
        "relevant_ipc_sections": [],
        "relevant_sections": ["16"],
        "relevant_keywords": [
            "CGST Act",
            "Section 16",
            "input tax credit",
            "tax invoice",
            "received goods",
            "tax paid to government",
            "return filed",
            "180 days",
        ],
        "expected_acts": ["Central Goods and Services Tax Act 2017"],
        "reference_answer": (
            "Section 16 of the CGST Act 2017 allows a registered person to claim "
            "input tax credit on goods/services used in the course of business, "
            "subject to conditions: possession of a valid tax invoice, actual "
            "receipt of the goods or services, the supplier having actually paid "
            "the tax to the government and furnished the invoice in their "
            "returns (reflected in GSTR-2B), and the recipient having filed "
            "their return. Payment to the supplier must be made within 180 days, "
            "failing which the credit is reversed. Credit cannot be claimed "
            "after the statutory cut-off date for the financial year."
        ),
        "domain": "tax_law",
    },
    {
        "query": "Who is required to file an income tax return in India and what happens on late filing?",
        "relevant_ipc_sections": [],
        "relevant_sections": ["139", "234F"],
        "relevant_keywords": [
            "Income Tax Act",
            "Section 139",
            "return of income",
            "due date",
            "belated return",
            "Section 234F",
            "late fee",
        ],
        "expected_acts": ["Income Tax Act 1961"],
        "reference_answer": (
            "Section 139(1) of the Income Tax Act 1961 requires every person "
            "whose total income exceeds the basic exemption limit (and companies "
            "and firms regardless of income) to file a return by the due date. "
            "A belated return may be filed under Section 139(4) before three "
            "months prior to the end of the assessment year, but late filing "
            "attracts a fee under Section 234F (up to INR 5,000) and interest "
            "under Section 234A, and certain losses cannot be carried forward."
        ),
        "domain": "tax_law",
    },
    # ── environment ─────────────────────────────────────────────────────────
    {
        "query": "What powers does the Central Government have under the Environment Protection Act to control pollution?",
        "relevant_ipc_sections": [],
        "relevant_sections": ["3", "5"],
        "relevant_keywords": [
            "Environment Protection Act 1986",
            "Section 3",
            "Section 5",
            "directions",
            "closure",
            "standards",
            "hazardous substances",
        ],
        "expected_acts": ["Environment Protection Act 1986"],
        "reference_answer": (
            "Under Section 3 of the Environment (Protection) Act 1986, the "
            "Central Government may take all measures necessary to protect and "
            "improve environmental quality — laying down emission/effluent "
            "standards, restricting industrial locations, and handling hazardous "
            "substances. Under Section 5 it can issue binding written directions "
            "to any person or authority, including orders to close, prohibit or "
            "regulate an industry, or stop/regulate its electricity or water "
            "supply. Contravention is punishable under Section 15 with "
            "imprisonment up to five years and/or fine."
        ),
        "domain": "environment_law",
    },
    # ── consumer ────────────────────────────────────────────────────────────
    {
        "query": "How do I file a consumer complaint for a defective product and what relief can I get?",
        "relevant_ipc_sections": [],
        "relevant_sections": ["35", "2"],
        "relevant_keywords": [
            "Consumer Protection Act 2019",
            "Section 35",
            "District Commission",
            "defect",
            "refund",
            "compensation",
            "product liability",
            "e-filing",
        ],
        "expected_acts": ["Consumer Protection Act 2019"],
        "reference_answer": (
            "Under Section 35 of the Consumer Protection Act 2019, a consumer "
            "may file a complaint before the District Consumer Commission "
            "(claims up to INR 50 lakh; higher values go to the State or "
            "National Commission), including electronically. The complaint can "
            "seek removal of the defect, replacement, refund of the price, and "
            "compensation for loss or injury, including punitive damages. "
            "Product liability claims against manufacturers/sellers are covered "
            "by Chapter VI. No court fee-heavy civil suit is needed and appeals "
            "lie up the Commission hierarchy."
        ),
        "domain": "consumer_law",
    },
    # ── cyber / IP ──────────────────────────────────────────────────────────
    {
        "query": "What is the punishment for identity theft and cheating by personation online?",
        "relevant_ipc_sections": [],
        "relevant_sections": ["66C", "66D"],
        "relevant_keywords": [
            "Information Technology Act",
            "Section 66C",
            "Section 66D",
            "identity theft",
            "personation",
            "password",
            "electronic signature",
            "three years",
        ],
        "expected_acts": ["Information Technology Act 2000"],
        "reference_answer": (
            "Section 66C of the IT Act 2000 punishes identity theft — fraudulent "
            "or dishonest use of another person's electronic signature, password "
            "or unique identification feature — with imprisonment up to three "
            "years and fine up to INR 1 lakh. Section 66D punishes cheating by "
            "personation using a computer resource (e.g., phishing, fake "
            "profiles) with the same maximum term and fine. The corresponding "
            "IPC cheating provisions can be invoked alongside."
        ),
        "domain": "cyber_law",
    },
    {
        "query": "What exclusive rights does a copyright owner have and how long does copyright protection last?",
        "relevant_ipc_sections": [],
        "relevant_sections": ["14", "22"],
        "relevant_keywords": [
            "Copyright Act 1957",
            "Section 14",
            "exclusive right",
            "reproduction",
            "sixty years",
            "Section 22",
            "literary",
            "adaptation",
        ],
        "expected_acts": ["Copyright Act 1957"],
        "reference_answer": (
            "Section 14 of the Copyright Act 1957 grants the owner exclusive "
            "rights to reproduce the work, issue copies, perform or communicate "
            "it to the public, make cinematograph films or sound recordings, "
            "translations and adaptations. For literary, dramatic, musical and "
            "artistic works, Section 22 provides protection for the author's "
            "lifetime plus sixty years from the year following the author's "
            "death; for films, sound recordings, photographs and anonymous "
            "works, sixty years from publication."
        ),
        "domain": "ip_law",
    },
    # ── election ────────────────────────────────────────────────────────────
    {
        "query": "What constitutes a corrupt practice in Indian elections?",
        "relevant_ipc_sections": [],
        "relevant_sections": ["123"],
        "relevant_keywords": [
            "Representation of the People Act 1951",
            "Section 123",
            "corrupt practice",
            "bribery",
            "undue influence",
            "religion",
            "booth capturing",
            "election petition",
        ],
        "expected_acts": ["Representation of the People Act 1951"],
        "reference_answer": (
            "Section 123 of the Representation of the People Act 1951 defines "
            "corrupt practices: bribery; undue influence; appeal to vote on "
            "grounds of religion, race, caste, community or language; promotion "
            "of enmity between classes; publication of false statements about a "
            "candidate's character; free conveyance of voters; election expenses "
            "beyond limits; obtaining government servants' assistance; and booth "
            "capturing. A candidate found guilty in an election petition may "
            "have their election declared void under Section 100 and face "
            "disqualification."
        ),
        "domain": "election_law",
    },
    # ── property ────────────────────────────────────────────────────────────
    {
        "query": "Can a homebuyer get a refund with interest if the builder delays possession under RERA?",
        "relevant_ipc_sections": [],
        "relevant_sections": ["18"],
        "relevant_keywords": [
            "RERA",
            "Section 18",
            "delayed possession",
            "refund",
            "interest",
            "withdraw from project",
            "compensation",
            "promoter",
        ],
        "expected_acts": ["Real Estate (Regulation and Development) Act 2016"],
        "reference_answer": (
            "Yes. Under Section 18 of the Real Estate (Regulation and "
            "Development) Act 2016, if a promoter fails to complete or hand over "
            "possession per the agreement for sale, the allottee may withdraw "
            "from the project and claim a full refund with interest at the "
            "prescribed rate plus compensation. If the allottee chooses to stay "
            "in the project, the promoter must pay interest for every month of "
            "delay until possession. Claims are filed before the RERA Authority "
            "or adjudicating officer, with appeal to the Appellate Tribunal."
        ),
        "domain": "property_law",
    },
    # ── commercial / arbitration ────────────────────────────────────────────
    {
        "query": "On what grounds can an arbitral award be set aside by a court?",
        "relevant_ipc_sections": [],
        "relevant_sections": ["34"],
        "relevant_keywords": [
            "Arbitration and Conciliation Act 1996",
            "Section 34",
            "set aside",
            "public policy",
            "patent illegality",
            "incapacity",
            "invalid agreement",
            "three months",
        ],
        "expected_acts": ["Arbitration and Conciliation Act 1996"],
        "reference_answer": (
            "Section 34 of the Arbitration and Conciliation Act 1996 permits a "
            "court to set aside an arbitral award only on limited grounds: "
            "incapacity of a party; invalid arbitration agreement; lack of "
            "proper notice or inability to present one's case; the award "
            "dealing with disputes beyond the submission; improper tribunal "
            "composition; the subject matter not being arbitrable; or conflict "
            "with the public policy of India (fraud, corruption, contravention "
            "of fundamental policy of Indian law, basic morality or justice). "
            "For domestic awards, patent illegality on the face of the award is "
            "an added ground. The application must be filed within three months "
            "(extendable by 30 days)."
        ),
        "domain": "commercial_law",
    },
    # ── human rights ────────────────────────────────────────────────────────
    {
        "query": "What are the functions and powers of the National Human Rights Commission?",
        "relevant_ipc_sections": [],
        "relevant_sections": ["12", "13"],
        "relevant_keywords": [
            "Protection of Human Rights Act 1993",
            "Section 12",
            "Section 13",
            "NHRC",
            "inquiry",
            "suo motu",
            "civil court powers",
            "recommendations",
        ],
        "expected_acts": ["Protection of Human Rights Act 1993"],
        "reference_answer": (
            "Section 12 of the Protection of Human Rights Act 1993 empowers the "
            "NHRC to inquire suo motu or on petition into human-rights "
            "violations or negligence by public servants, intervene in court "
            "proceedings, visit jails, review legal safeguards and treaties, "
            "and promote human-rights literacy. Under Section 13 it has the "
            "powers of a civil court — summoning witnesses, requiring document "
            "discovery, and receiving evidence on affidavit. Its recommendations "
            "(compensation, prosecution) are not directly binding but carry "
            "persuasive force, and it may approach constitutional courts."
        ),
        "domain": "human_rights_law",
    },
]
