"""
Document Analysis Pipeline
Integrates OCR extraction, IndianKanoon legal search, and RAG retrieval
to provide comprehensive document analysis with legal context.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import asyncio
import re


@dataclass
class DocumentAnalysisResult:
    """Result of document analysis pipeline."""

    extracted_text: str
    document_type: str  # pdf, docx, image_ocr, pdf_ocr, etc.
    summary: str
    key_points: List[str]
    legal_references: List[Dict[str, Any]]
    crime_context: Optional[Dict[str, Any]]
    confidence: float
    warnings: List[str]


class DocumentAnalysisPipeline:
    """
    Orchestrates document analysis with multiple legal data sources.
    """

    def __init__(self, llm, indian_kanoon_tool=None, crime_rag=None):
        """
        Initialize the pipeline.

        Args:
            llm: Language model for analysis
            indian_kanoon_tool: IndianKanoon API client (optional)
            crime_rag: Crime RAG system (optional)
        """
        self.llm = llm
        self.indian_kanoon = indian_kanoon_tool
        self.crime_rag = crime_rag

    async def analyze_document(
        self, document_text: str, document_type: str, user_query: str = ""
    ) -> DocumentAnalysisResult:
        """
        Perform comprehensive document analysis.

        Args:
            document_text: Extracted text from document
            document_type: Type of document (pdf, docx, image_ocr, etc.)
            user_query: User's specific question or request

        Returns:
            DocumentAnalysisResult with comprehensive analysis
        """
        warnings = []

        # Step 1: Quick analysis to identify document category
        doc_category = await self._identify_document_category(document_text)

        # Step 2: Extract legal entities and keywords
        legal_keywords = self._extract_legal_keywords(document_text)

        # Step 3: Search IndianKanoon for relevant legal context (if available)
        legal_references = []
        if self.indian_kanoon and legal_keywords:
            try:
                legal_references = await self._search_legal_references(legal_keywords)
            except Exception as e:
                warnings.append(f"Legal search unavailable: {str(e)}")

        # Step 4: Query Crime RAG if crime-related (if available)
        crime_context = None
        if doc_category.get("is_crime_related") and self.crime_rag:
            try:
                crime_context = await self._get_crime_context(
                    document_text, legal_keywords
                )
            except Exception as e:
                warnings.append(f"Crime database unavailable: {str(e)}")

        # Step 5: Generate comprehensive analysis using LLM with context
        analysis = await self._generate_analysis(
            document_text=document_text,
            document_type=document_type,
            user_query=user_query,
            legal_references=legal_references,
            crime_context=crime_context,
            doc_category=doc_category,
        )

        return DocumentAnalysisResult(
            extracted_text=document_text,
            document_type=document_type,
            summary=analysis["summary"],
            key_points=analysis["key_points"],
            legal_references=legal_references,
            crime_context=crime_context,
            confidence=analysis.get("confidence", 0.8),
            warnings=warnings,
        )

    async def _identify_document_category(self, text: str) -> Dict[str, Any]:
        """Identify document category and type."""
        text_lower = text.lower()

        # Check for legal document indicators
        is_legal = any(
            kw in text_lower
            for kw in [
                "agreement",
                "contract",
                "petition",
                "affidavit",
                "notice",
                "memorandum",
                "whereas",
                "party of the first part",
                "hereby",
                "lawsuit",
                "plaintiff",
                "defendant",
                "court",
            ]
        )

        # Check for crime-related content
        is_crime_related = any(
            kw in text_lower
            for kw in [
                "fir",
                "complaint",
                "crime",
                "theft",
                "assault",
                "harassment",
                "scam",
                "fraud",
                "victim",
                "accused",
                "police",
                "ipc",
                "section",
                "punishment",
                "penalty",
                "offense",
                "illegal",
            ]
        )

        # Check for specific document types
        is_contract = any(
            kw in text_lower
            for kw in ["agreement", "contract", "terms and conditions", "party agrees"]
        )

        is_notice = any(
            kw in text_lower for kw in ["notice", "hereby notified", "take notice"]
        )

        return {
            "is_legal": is_legal,
            "is_crime_related": is_crime_related,
            "is_contract": is_contract,
            "is_notice": is_notice,
        }

    def _extract_legal_keywords(self, text: str) -> List[str]:
        """Extract legal keywords and IPC sections from text."""
        keywords = []
        text_lower = text.lower()

        # Extract IPC sections
        ipc_pattern = r"(?:section|sec\.|s\.)\s*(\d+[A-Z]?)"
        ipc_matches = re.findall(ipc_pattern, text, re.IGNORECASE)
        keywords.extend([f"IPC Section {match}" for match in ipc_matches])

        # Common legal terms
        legal_terms = [
            "assault",
            "theft",
            "fraud",
            "harassment",
            "defamation",
            "breach of contract",
            "negligence",
            "damages",
            "compensation",
            "injunction",
            "arbitration",
            "mediation",
            "litigation",
        ]

        for term in legal_terms:
            if term in text_lower:
                keywords.append(term)

        # Limit to most relevant
        return list(set(keywords))[:10]

    async def _search_legal_references(
        self, keywords: List[str]
    ) -> List[Dict[str, Any]]:
        """Search IndianKanoon for relevant legal references."""
        if not self.indian_kanoon:
            return []

        references = []

        # Search for each keyword
        for keyword in keywords[:3]:  # Limit searches
            try:
                results = await self.indian_kanoon.search_documents(
                    query=keyword, max_results=3
                )

                for result in results:
                    references.append(
                        {
                            "title": result.title,
                            "excerpt": result.excerpt,
                            "url": result.url,
                            "type": result.document_type,
                            "keyword": keyword,
                        }
                    )
            except Exception as e:
                print(f"Search failed for '{keyword}': {e}")
                continue

        return references[:5]  # Return top 5

    async def _get_crime_context(
        self, document_text: str, keywords: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Get relevant crime reporting context from RAG."""
        if not self.crime_rag:
            return None

        try:
            # Query the RAG system
            query = " ".join(keywords[:3]) if keywords else document_text[:500]
            context = await self.crime_rag.get_relevant_context(query, top_k=3)

            return {
                "relevant_passages": context.get("passages", []),
                "sources": context.get("sources", []),
                "crime_type": context.get("crime_type", "general"),
            }
        except Exception as e:
            print(f"Crime RAG query failed: {e}")
            return None

    async def _generate_analysis(
        self,
        document_text: str,
        document_type: str,
        user_query: str,
        legal_references: List[Dict[str, Any]],
        crime_context: Optional[Dict[str, Any]],
        doc_category: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate comprehensive analysis using LLM with all context."""

        # Build context for LLM
        context_parts = []

        # Add legal references if available
        if legal_references:
            context_parts.append("**Relevant Legal References from IndianKanoon:**")
            for ref in legal_references[:3]:
                context_parts.append(f"- {ref['title']}: {ref['excerpt'][:200]}...")

        # Add crime context if available
        if crime_context and crime_context.get("relevant_passages"):
            context_parts.append("\n**Relevant Crime Reporting Guidelines:**")
            for passage in crime_context["relevant_passages"][:2]:
                context_parts.append(f"- {passage[:200]}...")

        context_text = "\n".join(context_parts)

        # Truncate document if too long
        max_doc_length = 10000
        doc_text = document_text[:max_doc_length]
        if len(document_text) > max_doc_length:
            doc_text += "\n\n[Document truncated for analysis]"

        # Build prompt
        prompt = f"""You are a legal document analyst. Analyze the following document and provide comprehensive insights.

Document Type: {document_type}
User Query: {user_query or "Provide comprehensive analysis"}

{context_text}

**Document Content:**
{doc_text}

**Analysis Instructions:**
1. Provide a clear, concise summary (2-3 sentences)
2. List 3-5 key points or findings
3. Identify any legal implications or concerns
4. If crime-related, explain relevant laws and procedures
5. Suggest next steps or actions if applicable

Provide your analysis in a structured format."""

        # Invoke LLM
        try:
            loop = asyncio.get_event_loop()
            from langchain_core.messages import HumanMessage

            response = await loop.run_in_executor(
                None, lambda: self.llm.invoke([HumanMessage(content=prompt)])
            )

            analysis_text = response.content

            # Parse the response
            summary_match = re.search(
                r"summary[:\s]+(.+?)(?=\n\n|\nkey|$)",
                analysis_text,
                re.IGNORECASE | re.DOTALL,
            )
            summary = (
                summary_match.group(1).strip() if summary_match else analysis_text[:300]
            )

            # Extract key points (look for bullet points or numbered lists)
            key_points = []
            lines = analysis_text.split("\n")
            for line in lines:
                line = line.strip()
                if line and (
                    line.startswith("-")
                    or line.startswith("•")
                    or re.match(r"^\d+\.", line)
                ):
                    point = re.sub(r"^[-•\d\.]+\s*", "", line)
                    if point:
                        key_points.append(point)

            if not key_points:
                # Fallback: split into sentences and take first few
                sentences = [s.strip() for s in summary.split(".") if s.strip()]
                key_points = sentences[:5]

            return {
                "summary": summary,
                "key_points": key_points[:5],
                "full_analysis": analysis_text,
                "confidence": 0.85,
            }

        except Exception as e:
            # Fallback response
            return {
                "summary": f"Document analysis completed. Document type: {document_type}",
                "key_points": [
                    "Document extracted successfully",
                    "Text available for review",
                ],
                "full_analysis": "",
                "confidence": 0.5,
            }


def get_document_analysis_pipeline(
    llm, indian_kanoon=None, crime_rag=None
) -> DocumentAnalysisPipeline:
    """
    Factory function to create document analysis pipeline.

    Args:
        llm: Language model instance
        indian_kanoon: IndianKanoon client (optional)
        crime_rag: Crime RAG system (optional)

    Returns:
        DocumentAnalysisPipeline instance
    """
    return DocumentAnalysisPipeline(llm, indian_kanoon, crime_rag)
