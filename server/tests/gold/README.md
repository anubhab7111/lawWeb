# Gold sets

| File | Source | Licence |
|---|---|---|
| `crime_type.jsonl` | `facts` and `crime_type` from [IndianBailJudgments-1200](https://huggingface.co/datasets/SnehaDeshmukh/IndianBailJudgments-1200) (arXiv 2507.02506), crime types merged onto `crime_reporter` types (see `FAMILY_TYPES` in `app/tools/crime_reporter.py`); dev/test split by case-id hash | CC BY 4.0 |
| `grounding.jsonl` | Claims from the chatbot's own answers to the `app/metrics/ground_truth*.py` questions (collected with `eval_grounding.py collect`), each with the statute text it was judged against, labelled blind (supported by that evidence or not); dev/test split by question | Statute text: public law; labels: this project |
| `routing.jsonl` | 80 messages per intent, dev/test 40/40 by id hash. general_query and crime_report: ILSIC lay questions of 80 words or fewer (labelled blind by intent; stored as ids, text read from the corpus drive) plus the 8 distinct real `chat_messages` turns; non_legal: [CLINC150](https://huggingface.co/datasets/clinc/clinc_oos) test queries (stored as ids; card-fraud/lost-card/freeze-account intents excluded); find_lawyer and document_analysis: hand-written (`routing_synthetic.py`) except 3 ILSIC lawyer requests | ILSIC CC BY-NC-SA 4.0 (not committed); CLINC150 CC BY 3.0; rest: this project |

The ILSIC lay-forum check in `eval_crime_type.py` reads ILSIC from the corpus drive at run time and is not committed (ILSIC is CC BY-NC-SA 4.0).
