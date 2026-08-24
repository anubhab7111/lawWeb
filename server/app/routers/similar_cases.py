"""
Similar Case Search — upload an order/petition/notice, extract FIRAC, then
search the existing case-law + statute RAG stack plus live Indian Kanoon for
similar Supreme Court / High Court cases and relevant statute sections.

Auth is optional: a logged-in user's search is recorded for history, an
anonymous search just isn't. No new FAISS index, no new embedding model —
everything here composes existing tools (document_extractor, firac_extractor,
legal_retrieval, indian_kanoon).
"""

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlmodel import Session

from app.config import get_settings
from app.db.engine import get_session
from app.db.models import User
from app.deps.auth import get_current_user_optional
from app.deps.uploads import read_upload_within_limit
from app.tool_dispatch import invoke_indian_kanoon
from app.tools.document_extractor import get_document_extractor
from app.tools.firac_extractor import FiracExtraction, extract_firac
from app.tools.legal_retrieval import retrieve_case_law, retrieve_statutes

router = APIRouter(prefix="/api/similar-cases", tags=["similar-cases"])


class TextSearchRequest(BaseModel):
    text: str = Field(..., min_length=20, max_length=20000)


def _serialize_firac(f: FiracExtraction) -> dict:
    return {
        "facts": f.facts,
        "issues": f.issues,
        "rules": f.rules,
        "application": f.application,
        "conclusion": f.conclusion,
        "extractedEntities": f.extracted_entities,
        "legalDomainHint": f.legal_domain_hint,
    }


async def _run_search(document_text: str) -> dict:
    firac = await extract_firac(document_text)
    query = firac.search_query()

    context, parsed = await retrieve_statutes(query, k=6)
    try:
        supreme_court_cases = await retrieve_case_law(
            query, parsed, context.chunks, k=5, query_issues=firac.issues
        )
    except Exception as e:
        print(f"[SimilarCaseSearch] case law retrieval failed: {e}")
        supreme_court_cases = []

    high_court_cases_text = ""
    try:
        ik_result = await invoke_indian_kanoon(query, context_type="general")
        if ik_result.succeeded:
            high_court_cases_text = ik_result.context_text
    except Exception as e:
        print(f"[SimilarCaseSearch] Indian Kanoon lookup failed: {e}")

    return {
        "firac": _serialize_firac(firac),
        "similarSupremeCourtCases": [
            {
                "caseName": c.case_name,
                "citation": c.citation,
                "court": c.court,
                "date": c.date,
                "url": c.url,
                "facts": c.facts,
                "issues": c.issues,
                "holding": c.holding,
                "score": round(c.score, 3),
            }
            for c in supreme_court_cases
        ],
        "relevantHighCourtCasesText": high_court_cases_text,
        "relevantStatutes": [
            {
                "actName": c.act_name,
                "sectionNumber": c.section_number,
                "title": c.title,
                "text": c.text[:1500],
            }
            for c in context.chunks
        ],
    }


@router.post("/search")
async def search_upload(
    file: Optional[UploadFile] = File(default=None),
    text: Optional[str] = Form(default=None),
    current_user: Optional[User] = Depends(get_current_user_optional),
    session: Session = Depends(get_session),
):
    if file is None and not text:
        raise HTTPException(status_code=400, detail="Provide either a file or text")

    if file is not None:
        max_size = get_settings().max_document_size_mb * 1024 * 1024
        file_bytes = await read_upload_within_limit(file, max_size)
        extractor = get_document_extractor()
        document_text, _doc_type = await extractor.extract_text(
            file_bytes, file.filename or "document.txt"
        )
    else:
        document_text = text or ""

    if not document_text.strip():
        raise HTTPException(
            status_code=422,
            detail="Could not extract text from the file. Please ensure it contains readable text.",
        )

    result = await _run_search(document_text)

    if current_user is not None:
        from app.db.models import SimilarCaseSearch

        session.add(
            SimilarCaseSearch(
                user_id=current_user.id,
                uploaded_doc_summary=result["firac"]["facts"][:1000],
                firac=result["firac"],
                results=result,
            )
        )
        session.commit()

    return result


@router.post("/search-text")
async def search_text(
    body: TextSearchRequest,
    current_user: Optional[User] = Depends(get_current_user_optional),
    session: Session = Depends(get_session),
):
    result = await _run_search(body.text)

    if current_user is not None:
        from app.db.models import SimilarCaseSearch

        session.add(
            SimilarCaseSearch(
                user_id=current_user.id,
                uploaded_doc_summary=result["firac"]["facts"][:1000],
                firac=result["firac"],
                results=result,
            )
        )
        session.commit()

    return result
