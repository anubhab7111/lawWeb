"""Crime-report statute retrieval on ILSIC lay questions that cite IPC sections.

hit@k: any of the k returned sections (BNS mapped back to IPC by the official table) is cited.
offence share: returned sections from substantive penal Acts (not CrPC/BNSS/evidence/definitions).
BNS share: returned penal sections cited under the BNS (in force since 1 July 2024).

    python eval_crime_sections.py --split dev|test [--limit N]
"""
import argparse
import ast
import asyncio
import json
import lzma
import random
import re
from pathlib import Path

ILSIC = Path("/run/media/ushtro/anubhab_x9/lawweb/ilsic/Layman-new-dataset")
CITE = re.compile(r"Section\s+(\S+)\s+of\s+The Indian Penal Code", re.I)
K = 2


def rows(split: str, limit: int) -> list:
    out = []
    for line in lzma.open(ILSIC / f"FT-Layman-{split}.jsonl.xz", "rt"):
        r = json.loads(line)
        a = r["answer"]
        labels = a if isinstance(a, list) else (ast.literal_eval(a) if a.startswith("[") else [a])
        cited = {m.group(1).upper() for l in labels for m in [CITE.search(l)] if m}
        if cited:
            out.append({"text": " ".join(r["instruction"].split())[:2000], "cited": cited})
    random.Random(4).shuffle(out)
    return out[:limit]


async def main(split: str, limit: int) -> None:
    from app.tool_dispatch import RAG_TOOL_REGISTRY
    from app.tools.crime_reporter import classify_crime_type
    from app.tools.fact_statutes import numbers_for

    data = rows(split, limit)
    hits = offence = bns = total = 0
    for k, r in enumerate(data):
        crime = await classify_crime_type(r["text"])
        res = await RAG_TOOL_REGISTRY["crime_sections"](r["text"], crime_type=crime, k=K)
        matches = res.raw.ipc_sections if res.raw is not None else []
        got = {n for m in matches for n in (numbers_for(m.act_name, m.section) or [])}
        hits += bool(got & r["cited"])
        for m in matches:
            total += 1
            penal = "Procedural" not in " ".join(m.reasons)
            offence += penal
            bns += penal and m.act_name.startswith("Bharatiya Nyaya")
        if k % 25 == 24:
            print(f"  {k + 1}/{len(data)} hit@{K} {hits / (k + 1):.3f}", flush=True)
    print(f"[{split}] n={len(data)} hit@{K} {hits / len(data):.3f}  offence share {offence / max(total, 1):.3f}  "
          f"BNS share {bns / max(offence, 1):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--limit", type=int, default=200)
    a = ap.parse_args()
    asyncio.run(main(a.split, a.limit))
