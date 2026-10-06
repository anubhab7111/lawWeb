# Gold sets

| File | Source | Licence |
|---|---|---|
| `crime_type.jsonl` | `facts` and `crime_type` from [IndianBailJudgments-1200](https://huggingface.co/datasets/SnehaDeshmukh/IndianBailJudgments-1200) (arXiv 2507.02506), crime types merged onto `crime_reporter` types (see `BAIL_TO_TYPES` in `eval_crime_type.py`); dev/test split by case-id hash | CC BY 4.0 |

The ILSIC lay-forum check in `eval_crime_type.py` reads ILSIC from the corpus drive at run time and is not committed (ILSIC is CC BY-NC-SA 4.0).
