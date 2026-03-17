"""
Indian Kanoon API integration module.
Provides access to Indian legal codes, case law, and statutes through the Indian Kanoon API.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import asyncio
import aiohttp
from functools import lru_cache
import json
from urllib.parse import quote


@dataclass
class LegalDocument:
    """Represents a legal document from Indian Kanoon."""

    title: str
    doc_id: str
    excerpt: str
    url: str
    document_type: str  # "case", "statute", "article"
    relevance_score: float = 0.0


@dataclass
class CaseLawResult:
    """Detailed case law information."""

    title: str
    doc_id: str
    court: str
    date: str
    citation: str
    summary: str
    full_text: str
    url: str
    related_docs: List[str]


@dataclass
class StatuteResult:
    """Statute or legal code information."""

    title: str
    doc_id: str
    act_name: str
    section: str
    content: str
    url: str
    amendments: List[str]


class IndianKanoonClient:
    """
    Client for interacting with Indian Kanoon API.
    Provides search, retrieval, and analysis of Indian legal documents.
    """

    BASE_URL = "https://api.indiankanoon.org"

    def __init__(self, api_key: str):
        """
        Initialize Indian Kanoon client.

        Args:
            api_key: API key for Indian Kanoon (from .env file)
        """
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None
        self._cache = {}  # Simple in-memory cache

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session with request timeout."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=12, connect=5)
            self.session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Token {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=timeout,
            )
        return self.session

    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()

    async def search_documents(
        self,
        query: str,
        doc_type: Optional[str] = None,
        page_num: int = 0,
        max_results: int = 10,
    ) -> List[LegalDocument]:
        """
        Search for legal documents on Indian Kanoon.

        Args:
            query: Search query (case name, statute, keywords)
            doc_type: Filter by type ("judgments", "statutes", "articles")
            page_num: Page number for pagination
            max_results: Maximum number of results to return

        Returns:
            List of LegalDocument objects
        """
        # Check cache
        cache_key = f"search:{query}:{doc_type}:{page_num}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        session = await self._get_session()

        # Build search URL - Indian Kanoon API requires POST requests
        endpoint = f"{self.BASE_URL}/search/"
        params = {"formInput": query, "pagenum": page_num}

        if doc_type:
            params["doctype"] = doc_type

        try:
            # Use POST instead of GET - Indian Kanoon API requirement
            async with session.post(endpoint, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = self._parse_search_results(data, max_results)
                    self._cache[cache_key] = results
                    return results
                else:
                    print(f"Indian Kanoon API error: {response.status}")
                    return []
        except asyncio.TimeoutError:
            print("Indian Kanoon API timeout")
            return []
        except Exception as e:
            print(f"Error searching Indian Kanoon: {e}")
            return []

    def _parse_search_results(
        self, data: Dict, max_results: int
    ) -> List[LegalDocument]:
        """Parse search results from API response."""
        results = []

        # The Indian Kanoon API returns results in a specific format
        docs = data.get("docs", [])

        for i, doc in enumerate(docs[:max_results]):
            try:
                doc_id = doc.get("tid", "")
                title = doc.get("title", "Untitled")
                excerpt = doc.get("headline", "")

                # Determine document type
                doc_type = "case"
                if "act" in title.lower() or "section" in title.lower():
                    doc_type = "statute"
                elif "article" in title.lower():
                    doc_type = "article"

                result = LegalDocument(
                    title=title,
                    doc_id=str(doc_id),
                    excerpt=excerpt,
                    url=f"https://indiankanoon.org/doc/{doc_id}/",
                    document_type=doc_type,
                    relevance_score=1.0 - (i * 0.05),  # Simple relevance scoring
                )
                results.append(result)
            except Exception as e:
                print(f"Error parsing search result: {e}")
                continue

        return results

    async def get_document_details(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific document.

        Args:
            doc_id: Document ID from Indian Kanoon

        Returns:
            Dictionary with document details
        """
        # Check cache
        cache_key = f"doc:{doc_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        session = await self._get_session()
        endpoint = f"{self.BASE_URL}/doc/{doc_id}/"

        try:
            # Use POST for document retrieval as well
            async with session.post(endpoint) as response:
                if response.status == 200:
                    data = await response.json()
                    self._cache[cache_key] = data
                    return data
                else:
                    print(f"Error fetching document {doc_id}: {response.status}")
                    return None
        except Exception as e:
            print(f"Error getting document details: {e}")
            return None

    async def search_ipc_section(self, section: str) -> List[LegalDocument]:
        """
        Search for specific IPC (Indian Penal Code) section.

        Args:
            section: IPC section number (e.g., "302", "420", "376")

        Returns:
            List of relevant documents
        """
        # Format IPC section query
        query = f"IPC Section {section}"
        return await self.search_documents(
            query=query, doc_type="judgments", max_results=15
        )

    async def search_crpc_section(self, section: str) -> List[LegalDocument]:
        """
        Search for specific CrPC (Code of Criminal Procedure) section.

        Args:
            section: CrPC section number

        Returns:
            List of relevant documents
        """
        query = f"CrPC Section {section}"
        return await self.search_documents(
            query=query, doc_type="judgments", max_results=15
        )

    async def search_act(
        self, act_name: str, section: Optional[str] = None
    ) -> List[LegalDocument]:
        """
        Search for specific Act and optionally a section within it.

        Args:
            act_name: Name of the Act (e.g., "Indian Evidence Act", "Contract Act")
            section: Optional section number within the Act

        Returns:
            List of relevant documents
        """
        query = act_name
        if section:
            query += f" Section {section}"

        return await self.search_documents(
            query=query, doc_type="statutes", max_results=15
        )

    async def search_case_law(
        self,
        keywords: str,
        court: Optional[str] = None,
        max_results: int = 15,
    ) -> List[LegalDocument]:
        """
        Search for case law by keywords and optionally filter by court.

        Args:
            keywords: Search keywords
            court: Optional court name filter (e.g., "Supreme Court", "High Court")
            max_results: Maximum number of results to return

        Returns:
            List of relevant case law documents
        """
        query = keywords
        if court:
            query += f" {court}"

        return await self.search_documents(
            query=query, doc_type="judgments", max_results=max_results
        )

    async def get_related_documents(self, doc_id: str) -> List[LegalDocument]:
        """
        Get documents related to a specific document.

        Args:
            doc_id: Document ID

        Returns:
            List of related documents
        """
        # Indian Kanoon doesn't have a direct related docs API
        # We'll fetch the document and extract citations
        doc_details = await self.get_document_details(doc_id)

        if not doc_details:
            return []

        # Extract potential related case citations from the text
        # This is a simplified approach - in production, you'd want better parsing
        title = doc_details.get("title", "")

        # Search for documents with similar titles
        return await self.search_documents(query=title, max_results=5)

    def format_search_results(self, results: List[LegalDocument]) -> str:
        """
        Format search results into a readable string for LLM context.

        Args:
            results: List of search results

        Returns:
            Formatted string
        """
        if not results:
            return "No relevant documents found in Indian Kanoon."

        formatted = "## Relevant Legal Documents from Indian Kanoon:\n\n"

        for i, doc in enumerate(results, 1):
            formatted += f"### {i}. {doc.title}\n"
            formatted += f"**Type:** {doc.document_type.title()}\n"
            formatted += f"**Document ID:** {doc.doc_id}\n"

            if doc.excerpt:
                formatted += f"**Excerpt:** {doc.excerpt}\n"

            formatted += f"**URL:** {doc.url}\n"
            formatted += f"**Relevance:** {doc.relevance_score:.2f}\n\n"

        return formatted

    def format_document_details(self, doc_details: Dict[str, Any]) -> str:
        """
        Format detailed document information for LLM context.

        Args:
            doc_details: Document details dictionary

        Returns:
            Formatted string
        """
        if not doc_details:
            return "Document details not available."

        formatted = "## Legal Document Details:\n\n"

        # Basic info
        title = doc_details.get("title", "Untitled")
        formatted += f"**Title:** {title}\n\n"

        # Document text (truncated for token limits)
        doc_text = doc_details.get("doc", "")
        if doc_text:
            max_length = 5000
            if len(doc_text) > max_length:
                doc_text = doc_text[:max_length] + "\n\n[Document truncated...]"
            formatted += f"**Content:**\n{doc_text}\n\n"

        return formatted


class IndianKanoonTool:
    """
    High-level tool for integrating Indian Kanoon API with the legal chatbot.
    Provides intelligent search and retrieval of Indian legal documents.
    """

    def __init__(self, api_key: str):
        """Initialize the tool with API key."""
        self.client = IndianKanoonClient(api_key)
        self._initialized = False

    async def initialize(self):
        """Initialize the tool (lazy initialization)."""
        if not self._initialized:
            self._initialized = True

    async def answer_legal_query(
        self, query: str, context_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Answer a legal query using Indian Kanoon documents.

        Args:
            query: User's legal question
            context_type: Type of context needed ("ipc", "crpc", "case_law", "statute", "general")

        Returns:
            Dictionary with answer and supporting documents
        """
        await self.initialize()

        # Determine search strategy based on context type
        results = []

        if context_type == "ipc":
            # Extract section numbers from query
            import re

            sections = re.findall(r"\b(\d{3}[A-Z]?)\b", query)
            if sections:
                for section in sections[:3]:  # Limit to 3 sections
                    section_results = await self.client.search_ipc_section(section)
                    results.extend(section_results)
            else:
                results = await self.client.search_documents(
                    query + " IPC", max_results=10
                )

        elif context_type == "crpc":
            import re

            sections = re.findall(r"\b(\d{3}[A-Z]?)\b", query)
            if sections:
                for section in sections[:3]:
                    section_results = await self.client.search_crpc_section(section)
                    results.extend(section_results)
            else:
                results = await self.client.search_documents(
                    query + " CrPC", max_results=10
                )

        elif context_type == "case_law":
            results = await self.client.search_case_law(query, max_results=10)

        elif context_type == "constitution":
            # Constitutional queries: search case law for landmark rulings
            results = await self.client.search_case_law(
                query + " Constitution India fundamental rights", max_results=10
            )

        elif context_type == "statute":
            results = await self.client.search_documents(
                query, doc_type="statutes", max_results=10
            )

        else:  # general
            results = await self.client.search_documents(query, max_results=10)

        # Remove duplicates based on doc_id
        unique_results = []
        seen_ids = set()
        for result in results:
            if result.doc_id not in seen_ids:
                unique_results.append(result)
                seen_ids.add(result.doc_id)

        return {
            "query": query,
            "context_type": context_type,
            "results": unique_results[:10],  # Limit to top 10
            "formatted_results": self.client.format_search_results(unique_results[:10]),
            "total_found": len(unique_results),
        }

    async def get_punishment_details(self, crime_description: str) -> Dict[str, Any]:
        """
        Get punishment details for a specific crime from IPC and case law.

        Args:
            crime_description: Description of the crime

        Returns:
            Dictionary with punishment information
        """
        await self.initialize()

        # Search for relevant IPC sections
        query = f"{crime_description} punishment penalty IPC"
        results = await self.client.search_documents(query, max_results=15)

        return {
            "crime": crime_description,
            "results": results,
            "formatted_results": self.client.format_search_results(results),
            "sources": [doc.url for doc in results[:5]],
        }

    async def close(self):
        """Close the client connection."""
        await self.client.close()


# Singleton pattern for the tool
_indian_kanoon_tool: Optional[IndianKanoonTool] = None


def get_indian_kanoon_tool(api_key: Optional[str] = None) -> IndianKanoonTool:
    """
    Get or create the Indian Kanoon tool instance.

    Args:
        api_key: Optional API key (will use from config if not provided)

    Returns:
        IndianKanoonTool instance
    """
    global _indian_kanoon_tool

    if _indian_kanoon_tool is None:
        if api_key is None:
            from app.config import get_settings

            settings = get_settings()
            api_key = settings.indian_kanoon_api_key

        _indian_kanoon_tool = IndianKanoonTool(api_key)

    return _indian_kanoon_tool


async def search_indian_law(query: str, context_type: str = "general") -> str:
    """
    Convenience function to search Indian law documents.

    Args:
        query: Search query
        context_type: Type of legal context

    Returns:
        Formatted string with search results
    """
    tool = get_indian_kanoon_tool()
    result = await tool.answer_legal_query(query, context_type)
    return result["formatted_results"]
