"""
Legal Defect Analyzer (Layer 3) — ReAct Pattern
Uses Reasoning-then-Action (ReAct) prompting for document validation.

2-Step LLM Pipeline:
  STEP 1 — THINK:   LLM reasons about what the document type requires
                     (governing statutes, mandatory elements, formalities)
  STEP 2 — ANALYZE: LLM reads the actual document text, cross-checks each
                     requirement against the text AND the regex findings,
                     and writes the final defect report with Act/Section/Case
                     Law citations and remediation

IMPORTANT: This module NEVER provides binding legal opinions.
All output is framed as:
"Based on statutory requirements and standard drafting practices,
the following potential issues were identified…"
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple

from app.tools.document_classifier import DocumentClassification
from app.tools.statutory_validator import StatutoryValidationResult, format_score
from app.tools.indian_law_rag import IndianLawContext
from app.prompts import (
    REACT_ANALYZE_PROMPT,
    REACT_THINK_PROMPT,
    sanitize_untrusted_document,
)

# ============================================================================
# Disclaimer Templates
# ============================================================================

# THINK sees only the document type, sub-type and jurisdiction, never the document,
# so its output is reused for every later document of the same kind.
_THINK_CACHE: Dict[tuple, str] = {}

DISCLAIMER_HEADER = (
    "**⚠️ Disclaimer:** This analysis is for informational and educational purposes only. "
    "It does not constitute a binding legal opinion or professional legal advice. "
    "Please consult a qualified legal practitioner registered with the Bar Council of India "
    "for advice specific to your situation."
)

DISCLAIMER_FOOTER = (
    "---\n"
    "*This analysis is generated using a multi-step reasoning pipeline based on "
    "statutory requirements and standard drafting practices under applicable Indian law. "
    "It identifies potential formal defects and statutory non-compliance but does not assess "
    "the substantive validity or enforceability of the document. "
    "For a definitive legal opinion, consult a practising advocate.*"
)


class LegalDefectAnalyzer:
    """
    Layer 3: ReAct-based legal reasoning and defect explanation.

    Instead of a single monolithic LLM prompt, this analyzer uses a
    2-step Reasoning-then-Action (ReAct) pipeline:

    1. THINK   — LLM reasons about requirements (no document text yet)
    2. ANALYZE — LLM reads document + regex findings, cross-checks, and
                 produces the final report

    This dramatically improves accuracy because:
    - The LLM first builds an independent mental model of requirements
    - It then reads the document text directly (not just regex results)
    - It can catch false positives/negatives from the regex layer
    - Each step's output feeds into the next, creating a reasoning chain
    """

    def __init__(self, llm):
        """
        Initialize with an LLM instance.

        Args:
            llm: ChatOllama or compatible LLM instance
        """
        self.llm = llm

    async def analyze_defects(
        self,
        classification: DocumentClassification,
        validation: StatutoryValidationResult,
        law_context: IndianLawContext,
        document_text: str = "",
        think_output: Optional[str] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive legal defect analysis using the ReAct pipeline.

        Args:
            classification: Layer 1 document classification result
            validation: Layer 2 statutory validation result
            law_context: Indian law context from RAG tool
            document_text: Original document text (up to ~5000 chars)

        Returns:
            Dict with structured defect analysis, reasoning trace, and formatted response
        """
        reasoning_trace: Dict[str, str] = {}

        # ==================================================================
        # STEP 1: THINK — Reason about requirements
        # ==================================================================
        print("[ReAct] Step 1/2: THINK — Reasoning about requirements...")
        if think_output is None:
            think_output = await self.think(classification)
        reasoning_trace["think"] = think_output
        print(f"[ReAct] THINK complete ({len(think_output)} chars)")

        # ==================================================================
        # STEP 2: ANALYZE — Check the document and produce the final report.
        # (A separate OBSERVE call used to sit here; the small model spent its
        # whole token budget thinking through it and usually fell back.)
        # ==================================================================
        print("[ReAct] Step 2/2: ANALYZE — Producing final report...")
        if stream:
            from app.chatbot import emit_text

            await emit_text(self._format_prefix(classification, validation, reasoning_trace) + "\n")
        analyze_output = await self._step_analyze(
            classification, validation, law_context, think_output, document_text, stream=stream
        )
        analyze_output, removed = await self._strip_unverified_cases(analyze_output)
        if removed:
            print(f"[ReAct] removed unverified case names: {removed}")
        if stream:
            await emit_text("\n" + self._format_suffix(law_context))
        reasoning_trace["analyze"] = analyze_output
        print(f"[ReAct] ANALYZE complete ({len(analyze_output)} chars)")

        # Build the final formatted response
        formatted_response = self._format_final_response(
            classification, validation, law_context, analyze_output, reasoning_trace
        )

        return {
            "classification": classification.to_dict(),
            "validation": validation.to_dict(),
            "law_context": law_context.to_dict(),
            "llm_analysis": analyze_output,
            "reasoning_trace": reasoning_trace,
            "formatted_response": formatted_response,
            "compliance_score": validation.compliance_score,
            "defect_count": validation.failed,
            "document_type": classification.document_type,
        }

    # ======================================================================
    # ReAct Step Implementations
    # ======================================================================

    async def think(self, classification: DocumentClassification) -> str:
        """THINK step, cached per document kind; fallbacks are not cached."""
        key = (
            classification.document_type,
            classification.sub_type,
            tuple(classification.jurisdiction_hints or ()),
        )
        if key not in _THINK_CACHE:
            try:
                checklist = await self._step_think(classification)
                _THINK_CACHE[key], _ = await self._strip_unverified_cases(checklist)
            except Exception as e:
                print(f"[ReAct] THINK step error: {e}")
                return self._fallback_think(classification)
        return _THINK_CACHE[key]

    async def _step_think(self, classification: DocumentClassification) -> str:
        """
        STEP 1: THINK — LLM reasons about what the document requires.
        No document text is provided yet — purely reasoning from knowledge.
        """
        prompt = REACT_THINK_PROMPT.format(
            document_type=classification.document_type,
            sub_type=classification.sub_type or "N/A",
            jurisdiction=(
                ", ".join(classification.jurisdiction_hints)
                if classification.jurisdiction_hints
                else "None detected"
            ),
        )

        return await self._invoke_llm(prompt)

    @staticmethod
    def _regex_findings(validation: StatutoryValidationResult) -> Dict[str, str]:
        """Layer 2 findings as prompt text."""
        # Format present elements
        present_str = ""
        if validation.present_elements:
            present_items = []
            for item in validation.present_elements:
                present_items.append(
                    f"  ✅ {item['element']}: {item.get('description', '')}"
                )
            present_str = "\n".join(present_items)
        else:
            present_str = "  (none detected)"

        # Format missing elements
        missing_str = ""
        if validation.missing_elements:
            missing_items = []
            for item in validation.missing_elements:
                missing_items.append(
                    f"  ❌ {item['element']}: {item['description']} "
                    f"(Required by: {item['statute_reference']})"
                )
            missing_str = "\n".join(missing_items)
        else:
            missing_str = "  (none — all mandatory elements found by regex)"

        # Format non-compliance
        nc_str = ""
        if validation.non_compliance:
            nc_items = []
            for item in validation.non_compliance:
                nc_items.append(
                    f"  ⚠️ {item['element']}: {item['description']} "
                    f"({item['statute_reference']})"
                )
            nc_str = "\n".join(nc_items)
        else:
            nc_str = "  (none identified)"

        return {"present_elements": present_str, "missing_elements": missing_str, "non_compliance": nc_str}

    async def _step_analyze(
        self,
        classification: DocumentClassification,
        validation: StatutoryValidationResult,
        law_context: IndianLawContext,
        think_output: str,
        document_text: str,
        stream: bool = False,
    ) -> str:
        """
        STEP 2: ANALYZE — LLM checks the document against the THINK checklist and
        the regex findings, with law context, and writes the final defect report.
        """
        # Format law context
        acts_str = (
            ", ".join(law_context.applicable_acts)
            if law_context.applicable_acts
            else "Not determined"
        )
        sections_str = (
            "\n".join([f"  - {s}" for s in law_context.applicable_sections])
            if law_context.applicable_sections
            else "  Not determined"
        )
        precedents_str = (
            "\n".join([f"  - {p}" for p in law_context.precedent_notes])
            if law_context.precedent_notes
            else "  None available"
        )
        state_notes_str = (
            "\n".join([f"  - {n}" for n in law_context.state_specific_notes])
            if law_context.state_specific_notes
            else "  None"
        )

        # Format API references
        api_refs_str = ""
        if law_context.references:
            ref_items = []
            for ref in law_context.references[:5]:
                ref_items.append(
                    f"  - {ref.title} ({ref.act_name} {ref.section}): "
                    f"{ref.excerpt[:150]}"
                )
            api_refs_str = "\n".join(ref_items)
        else:
            api_refs_str = "  None retrieved."

        prompt = REACT_ANALYZE_PROMPT.format(
            document_type=classification.document_type,
            think_output=think_output,
            passed=validation.passed,
            total_checks=validation.total_checks,
            **self._regex_findings(validation),
            document_text=sanitize_untrusted_document(
                document_text[:6000] if document_text else "(no document text available)"
            ),
            applicable_acts=acts_str,
            applicable_sections=sections_str,
            precedents=precedents_str,
            state_notes=state_notes_str,
            api_refs=api_refs_str,
            compliance_score=format_score(validation.compliance_score),
        )

        try:
            return await self._invoke_llm(prompt, stream=stream)
        except Exception as e:
            print(f"[ReAct] ANALYZE step error: {e}")
            return self._fallback_analyze(classification, validation, law_context)

    # ======================================================================
    # LLM Invocation
    # ======================================================================

    @staticmethod
    async def _strip_unverified_cases(text: str) -> Tuple[str, List[str]]:
        """Only cases in the indexed landmark corpus may stay in the report; if
        the corpus is unavailable nothing can be verified, so every case goes."""
        from app.tools.case_citation_verifier import strip_unverified_cases
        from app.tools.case_law_rag import get_case_law_rag_system

        rag = get_case_law_rag_system()
        try:
            ready = rag.initialized or await rag.initialize()
        except Exception:
            ready = False
        return strip_unverified_cases(text, rag.find_case if ready else (lambda name: None))

    async def _invoke_llm(self, prompt: str, stream: bool = False) -> str:
        """Invoke the LLM with a prompt and return the response content. A give-up
        is not streamed: the caller falls back and the handler replaces the text."""
        from app.chatbot import _INCOMPLETE_GENERATION_NOTE, invoke_llm_safely

        text = await invoke_llm_safely(self.llm, prompt, stream=stream, notify_incomplete=False)
        if text == _INCOMPLETE_GENERATION_NOTE:
            raise RuntimeError("LLM did not finish this step")
        return text

    # ======================================================================
    # Fallback Methods (when LLM is unavailable)
    # ======================================================================

    def _fallback_think(self, classification: DocumentClassification) -> str:
        """Generate a basic THINK output without LLM."""
        return (
            f"**THOUGHT — Document Requirements:**\n\n"
            f"Document type: {classification.document_type}\n"
            f"Sub-type: {classification.sub_type or 'N/A'}\n\n"
            f"This document type requires statutory compliance verification "
            f"against applicable Indian law. Automated checklist validation "
            f"will be used as the primary source of requirements.\n\n"
            f"*Note: LLM reasoning was unavailable for this step. "
            f"Falling back to rule-based checklist.*"
        )

    def _fallback_analyze(
        self,
        classification: DocumentClassification,
        validation: StatutoryValidationResult,
        law_context: IndianLawContext,
    ) -> str:
        """Generate a basic ANALYZE output without LLM as final fallback."""
        parts = []

        parts.append(
            "Based on statutory requirements and standard drafting practices, "
            "the following potential issues were identified:\n"
        )

        if validation.missing_elements:
            parts.append("**Missing Mandatory Elements:**\n")
            for item in validation.missing_elements:
                parts.append(f"- **{item['element']}**: {item['description']}")
                parts.append(f"  *Required by: {item['statute_reference']}*\n")

        if validation.non_compliance:
            parts.append("\n**Non-Compliance Findings:**\n")
            for item in validation.non_compliance:
                parts.append(f"- **{item['element']}**: {item['description']}")
                parts.append(f"  *Reference: {item['statute_reference']}*\n")

        if not validation.missing_elements and not validation.non_compliance:
            parts.append(
                "No formal defects or statutory non-compliance were identified "
                "based on the available text. However, substantive validity "
                "requires professional legal review."
            )

        parts.append(
            "\n**Recommended Next Steps:**\n"
            "1. Consult a qualified legal practitioner for substantive review.\n"
            "2. Verify stamp duty and registration compliance with local authorities.\n"
            "3. Ensure all original signatures and attestations are in order."
        )

        return "\n".join(parts)

    # ======================================================================
    # Response Formatting
    # ======================================================================

    def _format_final_response(
        self,
        classification: DocumentClassification,
        validation: StatutoryValidationResult,
        law_context: IndianLawContext,
        llm_analysis: str,
        reasoning_trace: Dict[str, str],
    ) -> str:
        """Format the complete response with all layers and reasoning trace."""
        prefix = self._format_prefix(classification, validation, reasoning_trace)
        return f"{prefix}\n{llm_analysis}\n{self._format_suffix(law_context)}"

    def _format_prefix(
        self,
        classification: DocumentClassification,
        validation: StatutoryValidationResult,
        reasoning_trace: Dict[str, str],
    ) -> str:
        """Everything before the ANALYZE text, ending with its heading."""
        parts = []

        # Header with disclaimer
        parts.append(DISCLAIMER_HEADER)
        parts.append("")

        # Document Classification Summary
        parts.append("## 📄 Document Classification")
        parts.append(f"**Type:** {classification.document_type}")
        if classification.sub_type:
            parts.append(f"**Sub-type:** {classification.sub_type}")
        parts.append(f"**Classification Confidence:** {classification.confidence:.0%}")
        if classification.jurisdiction_hints:
            parts.append(
                f"**Jurisdiction:** {', '.join(classification.jurisdiction_hints)}"
            )
        parts.append("")

        # Compliance Overview
        score = validation.compliance_score
        emoji = (
            "➖"
            if score is None
            else "✅" if score >= 0.8 else "⚠️" if score >= 0.5 else "❌"
        )
        parts.append("## 📊 Statutory Compliance Overview")
        parts.append(
            f"**Automated Compliance Score:** {emoji} "
            f"{format_score(score)}"
        )
        parts.append(f"**Checks Performed:** {validation.total_checks}")
        parts.append(
            f"**Passed:** {validation.passed} | **Failed:** {validation.failed}"
        )
        parts.append(
            "*Note: This score is from the automated regex check. "
            "The ReAct analysis below may adjust this assessment.*"
        )
        parts.append("")

        # Reasoning Trace Summary (collapsed sections)
        if "think" in reasoning_trace:
            parts.append("<details>")
            parts.append(
                "<summary>🧠 <strong>Reasoning Trace — THINK Step</strong> "
                "(click to expand)</summary>"
            )
            parts.append("")
            parts.append(reasoning_trace["think"])
            parts.append("")
            parts.append("</details>")
            parts.append("")

        # Main Analysis (ANALYZE step output) follows the heading.
        parts.append("## ⚖️ Legal Analysis")
        return "\n".join(parts)

    def _format_suffix(self, law_context: IndianLawContext) -> str:
        """Everything after the ANALYZE text."""
        parts = [""]

        # Applicable Law
        if law_context.applicable_acts:
            parts.append("## 📚 Applicable Indian Law")
            for act in law_context.applicable_acts:
                parts.append(f"- {act}")
            parts.append("")

        if law_context.applicable_sections:
            parts.append("### Key Sections")
            for sec in law_context.applicable_sections:
                parts.append(f"- {sec}")
            parts.append("")

        # State-specific notes
        if law_context.state_specific_notes:
            parts.append("## 🏛️ State-Specific Notes")
            for note in law_context.state_specific_notes:
                parts.append(f"- {note}")
            parts.append("")

        # Precedents
        if law_context.precedent_notes:
            parts.append("## 📎 Relevant Case Law")
            for p in law_context.precedent_notes:
                parts.append(f"- {p}")
            parts.append("")

        # Footer disclaimer
        parts.append(DISCLAIMER_FOOTER)

        return "\n".join(parts)


def get_legal_defect_analyzer(llm) -> LegalDefectAnalyzer:
    """Factory function to create a LegalDefectAnalyzer."""
    return LegalDefectAnalyzer(llm)
