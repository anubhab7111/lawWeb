"""
case_firac_extractor.py — FIRAC extraction for indexed CASE LAW judgments,
the corpus-side counterpart to firac_extractor.py (which extracts FIRAC from
the user's UPLOADED document at query time).

A judgment differs from a petition/order in shape: the operative holding and
ratio decidendi usually sit in the final third of the text, after lengthy
recitals of facts and arguments, so a head-only truncation (as
firac_extractor.py uses for short uploaded documents) would systematically
cut off the holding. This sampler takes head + tail instead.

Used offline by generate_case_firac.py to backfill app/data/case_law/*.json;
never called on the request path.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_ollama import ChatOllama

from app.chatbot import get_llm, invoke_llm_safely, strip_reasoning_tags

_HEAD_CHARS = 4500
_TAIL_CHARS = 3500


def _sample_judgment_text(text: str) -> str:
    if len(text) <= _HEAD_CHARS + _TAIL_CHARS:
        return text
    return text[:_HEAD_CHARS] + "\n...[omitted]...\n" + text[-_TAIL_CHARS:]


@dataclass
class CaseFiracExtraction:
    facts: str = ""
    issues: List[str] = field(default_factory=list)
    rules: List[str] = field(default_factory=list)
    holding: str = ""
    ratio_decidendi: str = ""
    domain: str = ""
    raw_llm_output: str = ""

    def to_json_dict(self) -> dict:
        return {
            "facts": self.facts,
            "issues": self.issues,
            "rules": self.rules,
            "holding": self.holding,
            "ratio_decidendi": self.ratio_decidendi,
            "domain": self.domain,
        }


_PROMPT = """You are extracting a FIRAC (Facts, Issues, Rules, holding, Ratio \
decidendi) summary from an Indian court judgment so it can be used to find \
similar cases and be compared against other legal issues.

Case: {case_name}
Citation: {citation}

Respond with ONLY a JSON object, no other text, in exactly this shape:
{{
  "facts": "<2-4 sentence summary of the factual background>",
  "issues": ["<legal issue/question the court decided 1>", "<legal issue 2>", ...],
  "rules": ["<statute, section, or legal principle the court applied>", ...],
  "holding": "<1-3 sentences: what the court actually decided/ordered>",
  "ratio_decidendi": "<1-2 sentences: the binding legal principle established, distinct from the facts of this case>",
  "domain": "<one of: criminal, civil, constitutional, family, property, corporate, labour, other>"
}}

Do not fabricate facts not present in the text. If a field cannot be \
determined, use an empty string or empty list.

JUDGMENT TEXT (head and tail of a long judgment; the middle has been omitted):
{text}
"""


def _extract_json(raw: str) -> dict:
    raw = raw.strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM output")
    return json.loads(match.group(0))


async def extract_case_firac(
    case_name: str,
    citation: str,
    text: str,
    llm: Optional[ChatOllama] = None,
) -> CaseFiracExtraction:
    llm = llm or get_llm()
    sampled = _sample_judgment_text(text)
    prompt = _PROMPT.format(case_name=case_name, citation=citation, text=sampled)

    try:
        raw = await invoke_llm_safely(llm, prompt)
    except Exception as e:
        print(f"[CaseFiracExtractor] LLM call failed for {case_name!r}: {e}")
        return CaseFiracExtraction()

    cleaned = strip_reasoning_tags(raw or "")

    try:
        parsed = _extract_json(cleaned)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"[CaseFiracExtractor] failed to parse LLM output for {case_name!r}: {e}")
        return CaseFiracExtraction(raw_llm_output=raw)

    return CaseFiracExtraction(
        facts=str(parsed.get("facts", "")),
        issues=[str(i) for i in parsed.get("issues", []) if i],
        rules=[str(r) for r in parsed.get("rules", []) if r],
        holding=str(parsed.get("holding", "")),
        ratio_decidendi=str(parsed.get("ratio_decidendi", "")),
        domain=str(parsed.get("domain", "")),
        raw_llm_output=raw,
    )
