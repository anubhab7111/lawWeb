"""
FIRAC extraction for Similar Case Search: turns an uploaded order/petition/
notice into a structured Facts/Issues/Rules/Application/Conclusion summary
that downstream retrieval (case law + statutes) is queried with.

Uses the main LLM (not the fast/classification one) because extraction
quality directly gates every downstream retrieval result — this is the
single biggest quality risk in Similar Case Search (see the implementation
plan). Callers should show the extracted FIRAC to the user for confirmation
before trusting search results built on it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List

from app.chatbot import get_llm, invoke_llm_safely

_MAX_INPUT_CHARS = 6000  # keep the extraction prompt well inside the LLM's context budget


@dataclass
class FiracExtraction:
    facts: str = ""
    issues: List[str] = field(default_factory=list)
    rules: List[str] = field(default_factory=list)
    application: str = ""
    conclusion: str = ""
    extracted_entities: List[str] = field(default_factory=list)
    legal_domain_hint: str = ""
    raw_llm_output: str = ""

    def search_query(self) -> str:
        """Text to feed into statute/case-law retrieval — issues carry the
        most retrieval signal, entities disambiguate domain."""
        parts = self.issues + self.extracted_entities
        return " ".join(parts) if parts else self.facts[:500]


_PROMPT = """You are extracting a FIRAC (Facts, Issues, Rules, Application, \
Conclusion) summary from a legal document (order, petition, or notice) so it \
can be used to search for similar cases and relevant statutes.

Respond with ONLY a JSON object, no other text, in exactly this shape:
{{
  "facts": "<2-4 sentence summary of the factual background>",
  "issues": ["<legal issue 1>", "<legal issue 2>", ...],
  "rules": ["<statute or legal principle invoked, if identifiable>", ...],
  "application": "<1-3 sentences on how the rules were argued/applied to the facts>",
  "conclusion": "<1-2 sentences on the outcome or relief sought, if stated>",
  "extracted_entities": ["<court>", "<statute names>", "<key legal terms>"],
  "legal_domain_hint": "<one of: criminal, civil, constitutional, family, property, corporate, labour, other>"
}}

Do not fabricate facts not present in the text. If a field cannot be \
determined, use an empty string or empty list.

DOCUMENT TEXT:
{text}
"""


def _extract_json(raw: str) -> dict:
    raw = raw.strip()
    # Models sometimes wrap JSON in ```json fences despite instructions.
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM output")
    return json.loads(match.group(0))


async def extract_firac(document_text: str) -> FiracExtraction:
    text = document_text[:_MAX_INPUT_CHARS]
    prompt = _PROMPT.format(text=text)

    try:
        raw = await invoke_llm_safely(get_llm(), prompt)
    except Exception as e:
        print(f"[FiracExtractor] LLM call failed: {e}")
        return FiracExtraction(facts=text[:500])

    try:
        parsed = _extract_json(raw)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"[FiracExtractor] failed to parse LLM output as JSON: {e}")
        return FiracExtraction(facts=text[:500], raw_llm_output=raw)

    return FiracExtraction(
        facts=str(parsed.get("facts", "")),
        issues=[str(i) for i in parsed.get("issues", []) if i],
        rules=[str(r) for r in parsed.get("rules", []) if r],
        application=str(parsed.get("application", "")),
        conclusion=str(parsed.get("conclusion", "")),
        extracted_entities=[str(e) for e in parsed.get("extracted_entities", []) if e],
        legal_domain_hint=str(parsed.get("legal_domain_hint", "")),
        raw_llm_output=raw,
    )
