"""
Lawyer finder module.
Provides lawyer search and recommendations based on specialization and location.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import random


@dataclass
class Lawyer:
    """Lawyer information."""

    name: str
    specialization: str
    location: str
    contact: str
    email: str
    rating: float
    experience_years: int
    languages: List[str]
    bar_number: str
    consultation_fee: str


class LawyerFinder:
    """
    Lawyer search and recommendation system.
    In production, this would connect to a real lawyer directory API.
    """

    # Legal specializations
    SPECIALIZATIONS = [
        "Criminal Defense",
        "Family Law",
        "Personal Injury",
        "Immigration",
        "Corporate Law",
        "Real Estate",
        "Intellectual Property",
        "Employment Law",
        "Tax Law",
        "Estate Planning",
        "Civil Rights",
        "Environmental Law",
        "Bankruptcy",
        "Medical Malpractice",
        "Contract Law",
    ]

    # Sample lawyer database (in production, this would be a real database/API)
    SAMPLE_LAWYERS: List[Dict] = [
        {
            "name": "Sarah Johnson, Esq.",
            "specialization": "Criminal Defense",
            "location": "New York, NY",
            "contact": "(212) 555-0101",
            "email": "sjohnson@lawfirm.com",
            "rating": 4.8,
            "experience_years": 15,
            "languages": ["English", "Spanish"],
            "bar_number": "NY12345",
            "consultation_fee": "Free initial consultation",
        },
        {
            "name": "Michael Chen, Esq.",
            "specialization": "Immigration",
            "location": "Los Angeles, CA",
            "contact": "(310) 555-0202",
            "email": "mchen@immigrationlaw.com",
            "rating": 4.9,
            "experience_years": 12,
            "languages": ["English", "Mandarin", "Cantonese"],
            "bar_number": "CA67890",
            "consultation_fee": "$100 for first hour",
        },
        {
            "name": "Emily Rodriguez, Esq.",
            "specialization": "Family Law",
            "location": "Chicago, IL",
            "contact": "(312) 555-0303",
            "email": "erodriguez@familylaw.com",
            "rating": 4.7,
            "experience_years": 10,
            "languages": ["English", "Spanish"],
            "bar_number": "IL11111",
            "consultation_fee": "Free 30-minute consultation",
        },
        {
            "name": "David Williams, Esq.",
            "specialization": "Personal Injury",
            "location": "Houston, TX",
            "contact": "(713) 555-0404",
            "email": "dwilliams@injurylaw.com",
            "rating": 4.6,
            "experience_years": 20,
            "languages": ["English"],
            "bar_number": "TX22222",
            "consultation_fee": "No fee unless we win",
        },
        {
            "name": "Jennifer Park, Esq.",
            "specialization": "Corporate Law",
            "location": "San Francisco, CA",
            "contact": "(415) 555-0505",
            "email": "jpark@corporatelaw.com",
            "rating": 4.9,
            "experience_years": 18,
            "languages": ["English", "Korean"],
            "bar_number": "CA33333",
            "consultation_fee": "$250 for first hour",
        },
        {
            "name": "Robert Thompson, Esq.",
            "specialization": "Real Estate",
            "location": "Miami, FL",
            "contact": "(305) 555-0606",
            "email": "rthompson@realestatelaw.com",
            "rating": 4.5,
            "experience_years": 14,
            "languages": ["English", "Spanish", "Portuguese"],
            "bar_number": "FL44444",
            "consultation_fee": "Free initial consultation",
        },
        {
            "name": "Lisa Anderson, Esq.",
            "specialization": "Employment Law",
            "location": "Seattle, WA",
            "contact": "(206) 555-0707",
            "email": "landerson@employmentlaw.com",
            "rating": 4.8,
            "experience_years": 11,
            "languages": ["English"],
            "bar_number": "WA55555",
            "consultation_fee": "Free initial consultation",
        },
        {
            "name": "James Wilson, Esq.",
            "specialization": "Criminal Defense",
            "location": "Phoenix, AZ",
            "contact": "(602) 555-0808",
            "email": "jwilson@criminaldefense.com",
            "rating": 4.7,
            "experience_years": 22,
            "languages": ["English", "Spanish"],
            "bar_number": "AZ66666",
            "consultation_fee": "Free initial consultation",
        },
        {
            "name": "Maria Garcia, Esq.",
            "specialization": "Immigration",
            "location": "San Diego, CA",
            "contact": "(619) 555-0909",
            "email": "mgarcia@visalaw.com",
            "rating": 4.9,
            "experience_years": 16,
            "languages": ["English", "Spanish", "French"],
            "bar_number": "CA77777",
            "consultation_fee": "$75 for first consultation",
        },
        {
            "name": "Christopher Lee, Esq.",
            "specialization": "Intellectual Property",
            "location": "Boston, MA",
            "contact": "(617) 555-1010",
            "email": "clee@iplaw.com",
            "rating": 4.8,
            "experience_years": 13,
            "languages": ["English", "Mandarin"],
            "bar_number": "MA88888",
            "consultation_fee": "$200 for first hour",
        },
        {
            "name": "Amanda Brown, Esq.",
            "specialization": "Family Law",
            "location": "Denver, CO",
            "contact": "(303) 555-1111",
            "email": "abrown@familymatters.com",
            "rating": 4.6,
            "experience_years": 9,
            "languages": ["English"],
            "bar_number": "CO99999",
            "consultation_fee": "Free 30-minute consultation",
        },
        {
            "name": "Daniel Martinez, Esq.",
            "specialization": "Tax Law",
            "location": "Dallas, TX",
            "contact": "(214) 555-1212",
            "email": "dmartinez@taxlaw.com",
            "rating": 4.7,
            "experience_years": 17,
            "languages": ["English", "Spanish"],
            "bar_number": "TX10101",
            "consultation_fee": "$150 for first hour",
        },
        {
            "name": "Rachel Kim, Esq.",
            "specialization": "Estate Planning",
            "location": "Atlanta, GA",
            "contact": "(404) 555-1313",
            "email": "rkim@estateplan.com",
            "rating": 4.8,
            "experience_years": 14,
            "languages": ["English", "Korean"],
            "bar_number": "GA12121",
            "consultation_fee": "Free initial consultation",
        },
        {
            "name": "Thomas Moore, Esq.",
            "specialization": "Bankruptcy",
            "location": "Philadelphia, PA",
            "contact": "(215) 555-1414",
            "email": "tmoore@debtrelief.com",
            "rating": 4.5,
            "experience_years": 19,
            "languages": ["English"],
            "bar_number": "PA13131",
            "consultation_fee": "Free initial consultation",
        },
        {
            "name": "Nicole White, Esq.",
            "specialization": "Civil Rights",
            "location": "Washington, DC",
            "contact": "(202) 555-1515",
            "email": "nwhite@civilrights.com",
            "rating": 4.9,
            "experience_years": 21,
            "languages": ["English", "French"],
            "bar_number": "DC14141",
            "consultation_fee": "Free initial consultation",
        },
    ]

    # Keyword mappings to specializations
    SPECIALIZATION_KEYWORDS: Dict[str, List[str]] = {
        "Criminal Defense": [
            "criminal",
            "crime",
            "arrested",
            "charges",
            "felony",
            "misdemeanor",
            "dui",
            "drug",
            "assault",
            "theft",
        ],
        "Family Law": [
            "divorce",
            "custody",
            "child support",
            "alimony",
            "marriage",
            "adoption",
            "family",
            "domestic",
        ],
        "Personal Injury": [
            "injury",
            "accident",
            "car accident",
            "slip and fall",
            "hurt",
            "medical bills",
            "compensation",
            "negligence",
        ],
        "Immigration": [
            "visa",
            "immigration",
            "green card",
            "citizenship",
            "deportation",
            "asylum",
            "work permit",
            "naturalization",
        ],
        "Corporate Law": [
            "business",
            "corporation",
            "startup",
            "merger",
            "acquisition",
            "contract",
            "llc",
            "partnership",
        ],
        "Real Estate": [
            "property",
            "house",
            "real estate",
            "landlord",
            "tenant",
            "lease",
            "eviction",
            "mortgage",
        ],
        "Intellectual Property": [
            "patent",
            "trademark",
            "copyright",
            "ip",
            "invention",
            "brand",
            "trade secret",
        ],
        "Employment Law": [
            "employment",
            "fired",
            "wrongful termination",
            "discrimination",
            "harassment",
            "workplace",
            "labor",
            "wages",
        ],
        "Tax Law": ["tax", "irs", "audit", "tax debt", "tax fraud", "tax planning"],
        "Estate Planning": [
            "will",
            "trust",
            "estate",
            "inheritance",
            "probate",
            "power of attorney",
            "living will",
        ],
        "Civil Rights": [
            "civil rights",
            "discrimination",
            "police brutality",
            "constitutional",
            "voting rights",
            "equal rights",
        ],
        "Environmental Law": [
            "environmental",
            "pollution",
            "epa",
            "clean water",
            "hazardous waste",
        ],
        "Bankruptcy": [
            "bankruptcy",
            "debt",
            "foreclosure",
            "chapter 7",
            "chapter 11",
            "chapter 13",
            "creditors",
        ],
        "Medical Malpractice": [
            "medical malpractice",
            "doctor error",
            "hospital negligence",
            "misdiagnosis",
            "surgical error",
        ],
        "Contract Law": [
            "contract",
            "breach",
            "agreement",
            "dispute",
            "terms",
            "negotiation",
        ],
    }

    def detect_specialization(self, query: str) -> Optional[str]:
        """
        Detect the needed legal specialization from user query.

        Args:
            query: User's search query or description

        Returns:
            Detected specialization or None
        """
        query_lower = query.lower()

        # Score each specialization
        scores: Dict[str, int] = {}
        for spec, keywords in self.SPECIALIZATION_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                scores[spec] = score

        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]

        return None

    def search_lawyers(
        self,
        specialization: Optional[str] = None,
        location: Optional[str] = None,
        min_rating: float = 0.0,
        min_experience: int = 0,
        language: Optional[str] = None,
        limit: int = 5,
    ) -> List[Lawyer]:
        """
        Search for lawyers based on criteria.

        Args:
            specialization: Legal specialization required
            location: Preferred location
            min_rating: Minimum rating threshold
            min_experience: Minimum years of experience
            language: Required language
            limit: Maximum number of results

        Returns:
            List of matching lawyers
        """
        results = []

        for lawyer_data in self.SAMPLE_LAWYERS:
            # Filter by specialization
            if (
                specialization
                and lawyer_data["specialization"].lower() != specialization.lower()
            ):
                continue

            # Filter by location
            if location and location.lower() not in lawyer_data["location"].lower():
                continue

            # Filter by rating
            if lawyer_data["rating"] < min_rating:
                continue

            # Filter by experience
            if lawyer_data["experience_years"] < min_experience:
                continue

            # Filter by language
            if language:
                if not any(
                    language.lower() in lang.lower()
                    for lang in lawyer_data["languages"]
                ):
                    continue

            results.append(Lawyer(**lawyer_data))

        # Sort by rating (descending)
        results.sort(key=lambda x: x.rating, reverse=True)

        return results[:limit]

    def search_by_query(self, query: str, limit: int = 5) -> List[Lawyer]:
        """
        Search lawyers based on a natural language query.

        Args:
            query: Natural language search query
            limit: Maximum number of results

        Returns:
            List of matching lawyers
        """
        # Detect specialization from query
        specialization = self.detect_specialization(query)

        # Try to detect location from query
        location = self._detect_location(query)

        # Search with detected criteria
        results = self.search_lawyers(
            specialization=specialization, location=location, limit=limit
        )

        # If no results with specialization, return general results
        if not results and specialization:
            results = self.search_lawyers(limit=limit)

        return results

    def _detect_location(self, query: str) -> Optional[str]:
        """Detect location from query string."""
        # Common locations to detect
        locations = [
            "new york",
            "los angeles",
            "chicago",
            "houston",
            "phoenix",
            "philadelphia",
            "san antonio",
            "san diego",
            "dallas",
            "san francisco",
            "seattle",
            "denver",
            "boston",
            "atlanta",
            "miami",
            "washington",
        ]

        query_lower = query.lower()
        for loc in locations:
            if loc in query_lower:
                return loc

        return None

    def format_lawyer_results(self, lawyers: List[Lawyer]) -> str:
        """Format lawyer search results as readable text."""
        if not lawyers:
            return "No lawyers found matching your criteria. Please try broadening your search."

        result_parts = [f"## Found {len(lawyers)} Lawyer(s)\n"]

        for i, lawyer in enumerate(lawyers, 1):
            result_parts.append(
                f"""
### {i}. {lawyer.name}
- **Specialization:** {lawyer.specialization}
- **Location:** {lawyer.location}
- **Experience:** {lawyer.experience_years} years
- **Rating:** {'⭐' * int(lawyer.rating)} ({lawyer.rating}/5.0)
- **Languages:** {', '.join(lawyer.languages)}
- **Contact:** {lawyer.contact}
- **Email:** {lawyer.email}
- **Consultation:** {lawyer.consultation_fee}
- **Bar Number:** {lawyer.bar_number}
"""
            )

        result_parts.append(
            "\n*Note: Always verify lawyer credentials with your state bar association before engaging their services.*"
        )

        return "".join(result_parts)

    def get_specializations(self) -> List[str]:
        """Get list of available specializations."""
        return self.SPECIALIZATIONS.copy()


# Singleton instance
_finder: Optional[LawyerFinder] = None


def get_lawyer_finder() -> LawyerFinder:
    """Get or create the lawyer finder instance."""
    global _finder
    if _finder is None:
        _finder = LawyerFinder()
    return _finder
