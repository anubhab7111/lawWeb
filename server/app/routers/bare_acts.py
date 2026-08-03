"""
Bare Act Explorer — public (no auth), read-only reference lookup over the
existing unified statute + case-law RAG stack (app/tools/bare_act_explorer.py).
"""

from fastapi import APIRouter, Query

from app.tools.bare_act_explorer import explore_bare_act

router = APIRouter(prefix="/api/bare-acts", tags=["bare-acts"])


def _serialize(result) -> dict:
    return {
        "query": result.query,
        "isSectionLookup": result.is_section_lookup,
        "ambiguous": result.ambiguous,
        "matches": [
            {
                "actName": m.act_name,
                "sectionNumber": m.section_number,
                "title": m.title,
                "text": m.text,
                "domain": m.domain,
            }
            for m in result.matches
        ],
        "landmarkJudgments": [
            {
                "caseName": c.case_name,
                "citation": c.citation,
                "court": c.court,
                "date": c.date,
                "url": c.url,
            }
            for c in result.landmark_judgments
        ],
        "explanation": result.explanation,
    }


@router.get("/search")
async def search(
    q: str = Query(..., min_length=1, max_length=300),
    act: str = Query(default="", description="Optional act-name hint to disambiguate a bare section number"),
):
    result = await explore_bare_act(q, act_hint=act)
    return _serialize(result)


@router.get("/section/{act}/{section_number}")
async def get_section(act: str, section_number: str):
    result = await explore_bare_act(f"Section {section_number}", act_hint=act)
    return _serialize(result)
