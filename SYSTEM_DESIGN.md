# System Design - Law Education Platform with AI Legal Assistant

## Table of Contents
1. [Overview](#overview)
2. [Architecture Diagram](#architecture-diagram)
3. [System Components](#system-components)
4. [Technology Stack](#technology-stack)
5. [Data Flow](#data-flow)
6. [Database Schema](#database-schema)
7. [API Architecture](#api-architecture)
8. [AI/ML Pipeline](#aiml-pipeline)
9. [Document Validation Pipeline](#document-validation-pipeline)
10. [Security & Authentication](#security--authentication)
11. [Scalability & Performance](#scalability--performance)
12. [Deployment Strategy](#deployment-strategy)

---

## Overview

The Law Education Platform is a full-stack web application that provides:
- **AI-powered legal consultation chatbot** using RAG (Retrieval-Augmented Generation)
- **Lawyer directory and recommendation system**
- **Booking and appointment management**
- **Payment gateway integration** (Braintree/Razorpay)
- **User authentication and profile management**

### Key Features
- Real-time legal analysis based on Indian Penal Code (IPC) and Bhartiya Nyaya Sanhita (BNS)
- LangGraph-based AI workflow with intent classification and specialized handlers
- **3-Layer Document Validation Pipeline** — deterministic classification, rule-based statutory checklist validation, and LLM-powered legal defect analysis
- Local LLM inference using Ollama with custom fine-tuned model (mistral-indian-law)
- OCR support for image-based document analysis (Tesseract)
- Indian Kanoon API integration for case law and statute retrieval
- FAISS vector database for semantic search of crime reporting guidelines
- Full-featured lawyer marketplace with ratings and specialties

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  React + TypeScript Frontend (Vite)                      │   │
│  │  - UI Components (Radix UI + shadcn/ui)                  │   │
│  │  - State Management (React Hooks)                        │   │
│  │  - API Integration Layer                                 │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP/REST
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API GATEWAY LAYER                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Express.js + TypeScript Backend (Port 5001)             │   │
│  │  ├─ Authentication Middleware (JWT)                      │   │
│  │  ├─ CORS Configuration                                   │   │
│  │  ├─ Request Logging                                      │   │
│  │  └─ Route Controllers                                    │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            │                 │                 │
            ▼                 ▼                 ▼
┌───────────────────┐  ┌──────────────┐  ┌─────────────────────┐
│   MongoDB Atlas   │  │  FastAPI     │  │  Payment Gateways   │
│   ├─ Users        │  │  (Port 8000) │  │  ├─ Braintree       │
│   ├─ Lawyers      │  │              │  │  └─ Razorpay        │
│   └─ Bookings     │  │  LangGraph   │  └─────────────────────┘
└───────────────────┘  │  Chatbot     │
                       └──────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            │                 │                 │
            ▼                 ▼                 ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │  Ollama      │  │  FAISS       │  │  Indian      │
    │  Local LLM   │  │  Vector DB   │  │  Kanoon API  │
    │  (Mistral)   │  │  (Crime RAG) │  │              │
    └──────────────┘  └──────────────┘  └──────────────┘
```

---

## System Components

### 1. Frontend (Client)

**Location:** `/client/`

**Technology:** React 18 + TypeScript + Vite

**Key Components:**
- **App.tsx**: Main application router and state container
- **ChatBot.tsx**: AI-powered legal consultation interface
- **LawyerDirectory.tsx**: Browse and search lawyers
- **LawyerRecommendation.tsx**: AI-based lawyer suggestions
- **LawyerProfile.tsx**: Detailed lawyer information
- **PaymentGateway.tsx**: Integrated payment processing
- **SignIn/SignUp.tsx**: Authentication forms

**UI Library:** 
- Radix UI primitives
- shadcn/ui component collection
- Tailwind CSS for styling

### 2. Backend API Server (Express)

**Location:** `/server/src/`

**Technology:** Node.js + Express + TypeScript

**Port:** 5001

**Routes:**
- `/api/auth` - User authentication (JWT-based)
- `/api/lawyers` - Lawyer CRUD operations
- `/api/bookings` - Appointment management
- `/api/chat` - Proxy to AI analysis service

**Models:**
- `User.ts` - User schema (name, email, password, timestamps)
- `Lawyer.ts` - Lawyer profile schema (specialty, rating, hourly rate, etc.)
- `Booking.ts` - Appointment records

### 3. AI Analysis Service (FastAPI + LangGraph)

**Location:** `/server/app/`

**Technology:** Python + FastAPI + LangGraph + Ollama

**Port:** 8000

**Architecture:** LangGraph State Machine with Intent-Based Routing

#### Core Components:

1. **Intent Classifier**
   - Fast keyword-based pre-classification for common patterns
   - LLM fallback for ambiguous cases
   - Intents: `document_analysis`, `document_validation`, `crime_report`, `find_lawyer`, `general_query`, `non_legal`

2. **Document Analysis Handler**
   - OCR extraction for images (Tesseract)
   - PDF/DOCX text extraction
   - Integration with Indian Kanoon API for legal references
   - Crime RAG context for crime-related documents
   - Document Analysis Pipeline for comprehensive analysis

3. **Crime Report Handler**
   - RAG-based IPC section extraction
   - Punishment details retrieval from legal documents
   - Indian Kanoon API for serious crimes
   - Structured guidance with immediate steps and legal advice

4. **Lawyer Finder Handler**
   - Specialization-based search
   - Location filtering
   - Legal context from Indian Kanoon (optional)

5. **General Query Handler**
   - Indian Kanoon API integration for legal questions
   - RAG retrieval for additional context
   - Conversational response generation

6. **Document Validation Handler** *(NEW — 3-Layer Pipeline)*
   - **Layer 1 — Document Classification** (deterministic, temperature=0): Weighted regex pattern matching to identify document type (Sale Deed, FIR, Affidavit, POA, Rent Agreement, etc.) with confidence scoring, sub-type detection, and jurisdiction hints
   - **Layer 2 — Statutory Checklist Validation** (rule-based, no LLM): Hard checklist of mandatory/recommended elements per document type with Act/Section references. Severity levels: mandatory, recommended, best_practice. Additional non-compliance checks (e.g., first-person narration for affidavits, consideration in words for sale deeds)
   - **Layer 2.5 — Indian Law Context Retrieval** (RAG + API): Parallel retrieval of applicable Acts, Sections, state-specific stamp duty notes, key precedents from static mapping + Indian Kanoon API (live) + FAISS RAG (local)
   - **Layer 3 — Legal Defect Analysis** (LLM-based): Comprehensive defect explanation with Consequence/Authority/Case Law/Remediation for each missing/non-compliant element. Safety-first: **never says "this document is legally valid"**
   - Uses Indian English legal terminology throughout
   - Supports 13 document types with extensible architecture

#### LangGraph Workflow:
```
User Input → Intent Classification → Route to Handler → Generate Response → END
                    │
                    ├─→ Document Analysis
                    ├─→ Document Validation (3-Layer Pipeline)
                    ├─→ Crime Report
                    ├─→ Find Lawyer
                    ├─→ General Query
                    └─→ Non-Legal (Rejection)
```

### 4. Vector Database (FAISS)

**Location:** `/server/app/data/faiss_index/`

**Purpose:** Semantic search of crime reporting guidelines and legal procedures

**Technology:** FAISS with persistent storage (LangChain integration)

**Embedding Model:** Ollama Embeddings (via langchain_ollama)

**Data Sources:**
- `crime_reporting_guide.txt` - Crime reporting procedures and legal guidance
- Indian legal statutes and IPC sections

### 5. Indian Kanoon API Integration

**Purpose:** Real-time legal case law and statute search

**Features:**
- Document search with relevance scoring
- Case law retrieval with citations
- Statute lookup with amendments
- Punishment details extraction

**Data Classes:**
- `LegalDocument` - Search result representation
- `CaseLawResult` - Detailed case information
- `StatuteResult` - Statute and legal code information

### 5. Database (MongoDB Atlas)

**Provider:** MongoDB Atlas (Cloud)

**Collections:**
- `users` - User authentication and profiles
- `lawyers` - Lawyer directory
- `bookings` - Appointment records

**Connection:** Mongoose ODM

---

## Technology Stack

### Frontend
| Layer | Technology | Version |
|-------|-----------|---------|
| Framework | React | 18.3.1 |
| Language | TypeScript | Latest |
| Build Tool | Vite | Latest |
| UI Components | Radix UI + shadcn/ui | Latest |
| Styling | Tailwind CSS | via class-variance-authority |
| HTTP Client | Fetch API | Native |
| Payment | Braintree Drop-in | 1.2.1 |

### Backend (Express)
| Layer | Technology | Version |
|-------|-----------|---------|
| Runtime | Node.js | 20+ |
| Framework | Express | 4.19.2 |
| Language | TypeScript | 5.4.3 |
| Database | MongoDB + Mongoose | 8.2.3 |
| Authentication | JWT (jsonwebtoken) | 9.0.3 |
| Password Hashing | bcryptjs | 3.0.3 |
| Payment | Braintree + Razorpay | 3.35.0, 2.9.6 |
| CORS | cors | 2.8.5 |

### AI Service (Python)
| Layer | Technology | Version |
|-------|-----------|--------|
| Framework | FastAPI | Latest |
| AI Orchestration | LangGraph | Latest |
| LLM Provider | Ollama (Local) | Latest |
| Model | mistral-indian-law:latest | Custom fine-tuned |
| Vector DB | FAISS | Latest |
| Embeddings | Ollama Embeddings | Via langchain_ollama |
| OCR | Tesseract + pytesseract | Latest |
| Document Processing | pdfplumber, pypdf, python-docx | Latest |
| Legal API | Indian Kanoon API | v1 |
| Validation | Pydantic | Latest |

---

## Data Flow

### 1. User Registration/Authentication Flow

```
Client                    Express API              MongoDB
  │                          │                       │
  ├─── POST /api/auth/signup ─────>                 │
  │                          │                       │
  │                     Hash password                │
  │                     (bcryptjs)                   │
  │                          │                       │
  │                          ├──── Create User ─────>│
  │                          │                       │
  │                          │<──── User Document ───┤
  │                          │                       │
  │                   Generate JWT                   │
  │                          │                       │
  │<─── Return token & user ─┤                       │
  │                          │                       │
```

### 2. Chat/Legal Analysis Flow

```
Client              Express API          FastAPI/LangGraph      FAISS/IK API
  │                     │                      │                     │
  ├─ POST /api/chat ───>│                      │                     │
  │                     │                      │                     │
  │                     ├─ POST /chat ────────>│                     │
  │                     │                      │                     │
  │                     │                      ├─ Classify Intent ──>│
  │                     │                      │                     │
  │                     │                      ├─ Route to Handler   │
  │                     │                      │    (based on intent)│
  │                     │                      │                     │
  │                     │                      ├─ [If crime_report] ─┼─ Query FAISS RAG
  │                     │                      │                     │
  │                     │                      ├─ [If general_query] ┼─ Query Indian Kanoon
  │                     │                      │                     │
  │                     │                      │<── Context ─────────┤
  │                     │                      │                     │
  │                     │                      ├─ Ollama LLM ───────>│
  │                     │                      │    (Generate Response)
  │                     │                      │                     │
  │                     │<── Response ─────────┤                     │
  │                     │                      │                     │
  │<─── Response ───────┤                      │                     │
  │                     │                      │                     │
```

### 3. Lawyer Booking Flow

```
Client              Express API          MongoDB          Payment Gateway
  │                     │                   │                    │
  ├─ Select Lawyer ────>│                   │                    │
  │                     │                   │                    │
  ├─ POST /api/bookings ─>                  │                    │
  │                     │                   │                    │
  │                Validate user            │                    │
  │                (JWT middleware)         │                    │
  │                     │                   │                    │
  │                     ├─ Create Booking ─>│                    │
  │                     │                   │                    │
  │<─── Payment UI ─────┤                   │                    │
  │                     │                   │                    │
  ├─ Process Payment ───┼───────────────────┼─────────────────>│
  │                     │                   │                    │
  │<─── Confirmation ───┼───────────────────┼────────────────────┤
  │                     │                   │                    │
  │                     ├─ Update Booking ─>│                    │
  │                     │   (status: paid)  │                    │
  │                     │                   │                    │
  │<─── Success ────────┤                   │                    │
  │                     │                   │                    │
```

---

## Database Schema

### Users Collection

```typescript
{
  _id: ObjectId,
  name: string,
  email: string,              // unique index
  password: string,            // bcrypt hashed
  createdAt: Date
}
```

### Lawyers Collection

```typescript
{
  _id: ObjectId,
  name: string,
  specialty: string,           // e.g., "Criminal Law", "Corporate Law"
  experience: number,          // years
  rating: number,              // 0-5
  hourlyRate: number,          // INR
  location: string,
  bio: string,
  cases: number,               // total cases handled
  successRate: number,         // percentage
  education: string,
  languages: string[],
  availability: string         // e.g., "Mon-Fri 9am-5pm"
}
```

### Bookings Collection

```typescript
{
  _id: ObjectId,
  userId: ObjectId,            // ref: User
  lawyerId: ObjectId,          // ref: Lawyer
  appointmentDate: Date,
  status: string,              // "pending", "confirmed", "paid", "completed"
  amount: number,
  paymentId: string,           // from payment gateway
  createdAt: Date,
  updatedAt: Date
}
```

### Vector Database (FAISS)

**Location:** `/server/app/data/faiss_index/`

**Index Structure:**
```python
# FAISS index with LangChain integration
{
  "index": faiss.IndexFlatL2,  # L2 distance for similarity
  "docstore": InMemoryDocstore,  # Document storage
  "index_to_docstore_id": dict,  # ID mapping
}
```

**Document Format:**
```python
# LangChain Document objects
{
  "page_content": "Crime reporting procedure text...",
  "metadata": {
    "source": "crime_reporting_guide.txt",
    "chunk_id": 0,
    "section": "Theft Reporting"
  }
}
```

---

## API Architecture

### REST API Endpoints

#### Authentication Routes (`/api/auth`)

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/signup` | Create new user | No |
| POST | `/signin` | Authenticate user | No |
| GET | `/profile` | Get user profile | Yes |
| PUT | `/profile` | Update profile | Yes |

**Request/Response Example:**

```typescript
// POST /api/auth/signup
Request: {
  name: string,
  email: string,
  password: string
}

Response: {
  token: string,
  user: {
    id: string,
    name: string,
    email: string
  }
}
```

#### Lawyer Routes (`/api/lawyers`)

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/` | List all lawyers | No |
| GET | `/:id` | Get lawyer by ID | No |
| POST | `/` | Create lawyer (admin) | Yes |
| PUT | `/:id` | Update lawyer | Yes |
| DELETE | `/:id` | Delete lawyer | Yes |
| GET | `/search?specialty=...` | Search lawyers | No |

**Response Example:**

```typescript
// GET /api/lawyers
Response: Lawyer[] = [
  {
    id: "...",
    name: "Advocate Sharma",
    specialty: "Criminal Law",
    experience: 15,
    rating: 4.8,
    hourlyRate: 3000,
    location: "Delhi",
    // ... other fields
  }
]
```

#### Booking Routes (`/api/bookings`)

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/user-bookings/:userId` | Get user bookings | Yes |
| POST | `/` | Create booking | Yes |
| PUT | `/:id` | Update booking | Yes |
| DELETE | `/:id` | Cancel booking | Yes |

#### Chat/Analysis Route (`/api/chat`)

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/` | Legal analysis | Optional |

**Request/Response Example:**

```typescript
// POST /api/chat
Request: {
  message: string
}

Response: {
  reply: {
    verified_crime: string,
    legal_verification: GroundedLegalConclusion,
    facts: IncidentFacts
  }
}
```

### Python AI Service API (FastAPI + LangGraph)

**Base URL:** `http://127.0.0.1:8000`

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serve frontend or health check |
| GET | `/health` | Health check endpoint |
| POST | `/chat` | Main chat endpoint |
| POST | `/chat/upload` | Chat with document/image upload |
| POST | `/analyze-document` | Analyze document text directly |
| POST | `/validate-document` | Validate document for statutory compliance (text) |
| POST | `/validate-document/upload` | Validate uploaded document for statutory compliance |
| POST | `/crime-report` | Get crime reporting guidance |
| POST | `/find-lawyer` | Search for lawyers |
| GET | `/specializations` | List legal specializations |
| GET | `/crime-types` | List recognized crime types |
| DELETE | `/session/{session_id}` | Clear chat session |
| GET | `/session/{session_id}/history` | Get session message history |

**Request/Response Example:**

```python
# POST /chat
Request: {
  "message": "I was assaulted yesterday, what should I do?",
  "session_id": "optional-session-id"
}

Response: {
  "response": "**Crime:** Assault\n\n**Statute:** Indian Penal Code (IPC) - Section 323/325 IPC...",
  "session_id": "uuid-string",
  "intent": "crime_report",
  "document_info": null,
  "crime_report": {
    "crime_type": "Assault",
    "severity": "Moderate",
    "immediate_steps": [...],
    "legal_steps": [...],
    "authorities_to_contact": [...],
    "evidence_to_preserve": [...],
    "timeline_advice": "...",
    "support_resources": [...]
  },
  "lawyers_found": null
}
```

**Document Validation Example:**

```python
# POST /validate-document
Request: {
  "document_text": "I, Ramesh Kumar, do hereby solemnly affirm...",
  "session_id": "optional-session-id"
}

Response: {
  "response": "## 📋 Document Classification\n**Type:** Affidavit...\n## ⚖️ Statutory Compliance...",
  "session_id": "uuid-string",
  "intent": "document_validation",
  "document_info": null,
  "document_validation": {
    "classified_type": "Affidavit",
    "classification_confidence": 0.87,
    "sub_type": "General Affidavit",
    "jurisdiction_hints": ["Delhi"],
    "compliance_score": 0.625,
    "total_checks": 8,
    "passed": 5,
    "failed": 3,
    "missing_elements": [
      {"element": "Notary attestation", "severity": "mandatory", "statute": "Notaries Act, 1952"}
    ],
    "present_elements": [
      {"element": "Deponent identity", "severity": "mandatory", "statute": "Indian Evidence Act, Section 3"}
    ],
    "non_compliance": ["Document not on stamp paper"],
    "llm_analysis": "### Detailed Analysis...\n...",
    "applicable_acts": ["Indian Evidence Act, 1872", "Oaths Act, 1969"],
    "applicable_sections": ["Section 3 - Interpretation clause", "Section 4 - Oaths"],
    "precedent_notes": ["Ghulam Qadir v. Special Tribunal (2002) - Defective affidavit..."],
    "state_specific_notes": ["Delhi: Stamp paper of Rs. 10/- required"]
  },
  "crime_report": null,
  "lawyers_found": null
}
```

**Document Upload Example:**

```python
# POST /chat/upload (multipart/form-data)
Request:
  file: <document.pdf or image.jpg>
  message: "Please analyze this document"
  session_id: "optional-session-id"

Response: {
  "response": "📄 **What is this?** Rental Agreement...\n\n📝 **Summary**...",
  "session_id": "uuid-string",
  "intent": "document_analysis",
  "document_info": {
    "text": "...",
    "summary": "...",
    "key_points": [...],
    "document_type": "pdf",
    "legal_references": [...],
    "confidence": 0.85
  },
  "crime_report": null,
  "lawyers_found": null
}
```

---

## AI/ML Pipeline

### LangGraph State Machine

**Core Files:**
- `app/chatbot.py` - LangGraph workflow definition
- `app/state.py` - State type definitions
- `app/prompts.py` - Prompt templates
- `app/config.py` - Configuration settings

### LangGraph Workflow

```python
# State Machine Flow
User Input
    ↓
[classify_intent]
    ├─ Keyword-based pre-classification
    ├─ LLM fallback for ambiguous cases
    └─ Output: intent (Literal)
    ↓
[route_by_intent] → Conditional Edge
    │
    ├──→ [handle_document_analysis]
    │       ├─ OCR extraction (if image)
    │       ├─ Indian Kanoon search
    │       ├─ Crime RAG context
    │       └─ DocumentAnalysisPipeline
    │
    ├──→ [handle_document_validation]  ← NEW
    │       ├─ Layer 1: DocumentClassifier (deterministic, temp=0)
    │       ├─ Layer 2: StatutoryValidator (rule-based, no LLM)
    │       ├─ Layer 2.5: IndianLawRAGTool (parallel async retrieval)
    │       ├─ Layer 3: LegalDefectAnalyzer (LLM-based reasoning)
    │       └─ Safety: NEVER says "this document is legally valid"
    │
    ├──→ [handle_crime_report]
    │       ├─ RAG-based IPC extraction
    │       ├─ Indian Kanoon (for serious crimes)
    │       ├─ Punishment details retrieval
    │       └─ Structured guidance generation
    │
    ├──→ [handle_find_lawyer]
    │       ├─ Specialization detection
    │       ├─ Location-based search
    │       └─ Optional legal context
    │
    ├──→ [handle_general_query]
    │       ├─ Indian Kanoon integration
    │       ├─ RAG retrieval
    │       └─ Conversational response
    │
    └──→ [handle_non_legal_query]
            └─ Polite rejection message
    ↓
Final Response (END)
```

### RAG (Retrieval-Augmented Generation)

**Implementation:** `app/tools/crime_rag.py`

**Components:**
- `CrimeRAG` - Main RAG system class
- `RuleLawTextSplitter` - Custom text splitter preserving legal structure
- `CrimeContext` - Retrieved context data class

**Process:**
1. Load legal documents from `/server/app/data/`
2. Split using structure-aware `RuleLawTextSplitter`
3. Generate embeddings using Ollama
4. Store in FAISS vector index
5. Retrieve top-k relevant passages for queries

**Advantages:**
- Local inference (no API costs)
- Custom fine-tuned model for Indian law
- Structure-preserving document splitting
- Fast semantic search with FAISS

### Indian Kanoon Integration

**Implementation:** `app/tools/indian_kanoon.py`

**Features:**
- Asynchronous API client with caching
- Document search with relevance scoring
- Punishment details extraction
- Case law and statute retrieval

**Usage:**
```python
client = IndianKanoonClient(api_key)
docs = await client.search_documents(query, max_results=10)
details = await client.get_document_details(doc_id)
```

### Document Analysis Pipeline

**Implementation:** `app/tools/document_analysis_pipeline.py`

**Steps:**
1. Identify document category (legal, crime-related, etc.)
2. Extract legal entities and keywords
3. Search Indian Kanoon for relevant references
4. Query Crime RAG if crime-related
5. Generate comprehensive analysis with LLM

### OCR Processing

**Implementation:** `app/tools/ocr_extractor.py`

**Supported Formats:**
- Images: JPG, JPEG, PNG, BMP, TIFF, GIF
- Scanned PDFs (via pdf2image conversion)

**Technology:** Tesseract OCR with pytesseract

### Data Models (Pydantic/TypedDict)

**ChatState:**
```python
class ChatState(TypedDict):
    messages: List[Message]
    current_input: str
    conversation_context: Optional[str]
    intent: Optional[Literal["document_analysis", "document_validation",
                             "crime_report", "find_lawyer", 
                             "general_query", "non_legal"]]
    document_content: Optional[str]
    document_type: Optional[str]
    document_info: Optional[DocumentInfo]
    document_validation: Optional[DocumentValidationInfo]
    crime_details: Optional[str]
    crime_report: Optional[CrimeReportInfo]
    lawyer_query: Optional[str]
    lawyers_found: Optional[List[LawyerInfo]]
    response: Optional[str]
    session_id: Optional[str]
    error: Optional[str]
```

**DocumentInfo:**
```python
class DocumentInfo(TypedDict):
    text: str
    summary: str
    key_points: List[str]
    document_type: str
    parties_involved: List[str]
    important_dates: List[str]
    legal_implications: str
```

**CrimeReportInfo:**
```python
class CrimeReportInfo(TypedDict):
    crime_type: str
    severity: str
    immediate_steps: List[str]
    legal_steps: List[str]
    authorities_to_contact: List[str]
    evidence_to_preserve: List[str]
    timeline_advice: str
    support_resources: List[str]
```

**DocumentValidationInfo:**
```python
class DocumentValidationInfo(TypedDict):
    # Layer 1 — Classification
    classified_type: str           # e.g., "Affidavit", "Sale Deed", "FIR"
    classification_confidence: float  # 0.0–0.99
    sub_type: Optional[str]        # e.g., "General Affidavit", "Judicial Affidavit"
    jurisdiction_hints: List[str]  # e.g., ["Delhi", "Maharashtra"]

    # Layer 2 — Statutory Validation
    compliance_score: float        # 0.0–1.0
    total_checks: int
    passed: int
    failed: int
    missing_elements: List[Dict[str, str]]   # [{element, severity, statute}]
    present_elements: List[Dict[str, str]]   # [{element, severity, statute}]
    non_compliance: List[str]                # Additional rule violations

    # Layer 3 — LLM Analysis
    llm_analysis: Optional[str]    # Markdown-formatted defect explanation

    # Law Context
    applicable_acts: List[str]
    applicable_sections: List[str]
    precedent_notes: List[str]
    state_specific_notes: List[str]
```

---

## Document Validation Pipeline

The document validation system uses a **3-layer architecture** designed to provide comprehensive statutory compliance analysis without ever rendering binding legal opinions.

> ⚠️ **Safety Principle:** The system will **NEVER** say "this document is legally valid". It can only identify potential deficiencies and missing elements.

### Architecture Overview

```
Document Text Input
        │
        ▼
┌─────────────────────────────────────┐
│  LAYER 1: Document Classification   │
│  (Deterministic — temperature=0)    │
│                                     │
│  • Weighted regex pattern matching  │
│  • Primary indicators (3x weight)   │
│  • Secondary indicators (1x weight) │
│  • Sub-type detection               │
│  • Jurisdiction hint extraction     │
│  • Confidence score (capped 0.99)   │
│                                     │
│  Output: DocumentClassification     │
│  (type, confidence, sub_type,       │
│   matched_indicators, jurisdiction) │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  LAYER 2: Statutory Validation      │
│  (Rule-based — NO LLM)             │
│                                     │
│  • Per-document-type checklists     │
│  • Regex-based element detection    │
│  • Severity: mandatory/recommended/ │
│    best_practice                    │
│  • Act & Section references         │
│  • Non-compliance rule checks       │
│                                     │
│  Output: StatutoryValidationResult  │
│  (score, passed, failed, missing,   │
│   present, non_compliance, warnings)│
└──────────────┬──────────────────────┘
               │
               ▼ (parallel)
┌─────────────────────────────────────┐
│  LAYER 2.5: Indian Law Context      │
│  (RAG + API — async parallel)       │
│                                     │
│  ┌──────────┐ ┌──────────┐          │
│  │ Static   │ │ Indian   │          │
│  │ Law Map  │ │ Kanoon   │          │
│  │ (13 doc  │ │ API      │          │
│  │  types)  │ │ (live)   │          │
│  └────┬─────┘ └────┬─────┘          │
│       │            │                │
│  ┌────┴────────────┴─────┐          │
│  │   Merged Context      │          │
│  │   + FAISS RAG results │          │
│  └───────────────────────┘          │
│                                     │
│  Output: IndianLawContext           │
│  (references, acts, sections,       │
│   state_notes, precedents)          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  LAYER 3: Legal Defect Analysis     │
│  (LLM-based — mistral-indian-law)   │
│                                     │
│  • Builds structured prompt with    │
│    all Layer 1+2+2.5 findings       │
│  • Per-defect analysis:             │
│    - Consequence                    │
│    - Statutory Authority             │
│    - Relevant Case Law              │
│    - Remediation Steps              │
│  • Fallback if LLM fails            │
│  • Disclaimer header & footer       │
│                                     │
│  Output: Formatted Markdown Report  │
│  (classification, compliance,       │
│   defects, law context, disclaimer) │
└─────────────────────────────────────┘
```

### Supported Document Types

| # | Document Type | Sub-types | Key Statutes |
|---|--------------|-----------|-------------|
| 1 | Sale Deed | Absolute, Conditional, Gift-cum-Sale | Transfer of Property Act 1882, Registration Act 1908, Indian Stamp Act 1899 |
| 2 | FIR (First Information Report) | Cognizable, Zero FIR, e-FIR | CrPC Section 154, BNS |
| 3 | Affidavit | General, Judicial, Income, Name Change | Oaths Act 1969, Indian Evidence Act 1872 |
| 4 | Agreement to Sell | Residential, Commercial, Agricultural | Indian Contract Act 1872, RERA 2016, Specific Relief Act 1963 |
| 5 | Power of Attorney | General, Special, Irrevocable | Powers of Attorney Act 1882, Registration Act 1908 |
| 6 | Rent Agreement | Residential, Commercial | Rent Control Act (state-specific), Registration Act 1908 |
| 7 | Notice (CrPC/CPC) | Legal, Eviction, Demand, Reply | CPC Section 80, CrPC |
| 8 | Court Order / Judgment | Interim, Final, Consent | CPC, CrPC, Constitution of India |
| 9 | Will / Testament | Simple, Privileged, Codicil | Indian Succession Act 1925 |
| 10 | Partnership Deed | General, LLP, Reconstitution | Indian Partnership Act 1932, LLP Act 2008 |
| 11 | Bail Application | Regular, Anticipatory, Default | CrPC Sections 436-439, BNS |
| 12 | Complaint (CrPC) | Private, Consumer, Magistrate | CrPC Section 200 |
| 13 | Chargesheet | Police, Supplementary | CrPC Section 173, BNS |

### Statutory Checklists

Each document type has a comprehensive checklist of required elements. Each item specifies:
- **Element name** and description
- **Governing statute** (Act + Section)
- **Severity level**: `mandatory` (must-have), `recommended` (should-have), `best_practice` (nice-to-have)
- **Regex patterns** for automated detection

**Example — Affidavit Checklist:**

| # | Element | Statute | Severity |
|---|---------|---------|----------|
| 1 | Deponent identity | Indian Evidence Act, Section 3 | Mandatory |
| 2 | Solemn affirmation | Oaths Act, 1969 — Section 4 | Mandatory |
| 3 | Statement of facts | Indian Evidence Act, Section 3 | Mandatory |
| 4 | Verification clause | CPC Order XIX Rule 3 | Mandatory |
| 5 | Stamp paper | Indian Stamp Act, 1899 | Mandatory |
| 6 | Notary attestation | Notaries Act, 1952 | Mandatory |
| 7 | Date and place | Indian Evidence Act | Recommended |
| 8 | Deponent signature | Indian Evidence Act, Section 67 | Mandatory |

### State-Specific Notes

The system includes stamp duty guidance for major states:
- **Delhi** — Rs. 10/- stamp paper for affidavits, 6% stamp duty on sale deeds
- **Maharashtra** — Rs. 100-500 stamp paper, 5% stamp duty in Mumbai, 6% outside
- **Karnataka** — Rs. 20/- stamp paper, 5.6% stamp duty
- **Tamil Nadu** — Rs. 20/- stamp paper, 7% stamp duty
- **Uttar Pradesh** — Rs. 10/- stamp paper, 7% stamp duty (M), 6% (F)
- **West Bengal** — Rs. 10/- stamp paper, 6% stamp duty in Kolkata, 7% outside
- **Rajasthan, Gujarat** — Similar state-specific rules

### Implementation Files

| File | Purpose | Layer |
|------|---------|-------|
| `app/tools/document_classifier.py` | Weighted regex document classification | Layer 1 |
| `app/tools/statutory_validator.py` | Rule-based checklist validation | Layer 2 |
| `app/tools/indian_law_rag.py` | RAG + API context retrieval | Layer 2.5 |
| `app/tools/legal_defect_analyzer.py` | LLM-based defect explanation | Layer 3 |
| `app/chatbot.py` → `handle_document_validation()` | Orchestration handler | All |
| `app/state.py` → `DocumentValidationInfo` | State type definition | — |
| `app/prompts.py` → Validation prompts | Indian English prompts | Layer 3 |

---

## Security & Authentication

### JWT-Based Authentication

**Flow:**
1. User signs up/signs in
2. Server generates JWT with user ID
3. Token stored in `localStorage` on client
4. Token sent in `Authorization` header for protected routes
5. Express middleware validates token

**Implementation:**
```typescript
// Middleware
const authMiddleware = (req, res, next) => {
  const token = req.headers.authorization?.split(' ')[1];
  if (!token) return res.status(401).json({ error: 'Unauthorized' });
  
  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    req.userId = decoded.id;
    next();
  } catch (err) {
    res.status(401).json({ error: 'Invalid token' });
  }
};
```

### Password Security

- **Hashing Algorithm:** bcryptjs (salt rounds: 10)
- **Storage:** Never store plaintext passwords
- **Validation:** Compare hashed password on login

### CORS Configuration

```typescript
app.use(cors({
  origin: process.env.CLIENT_URL || 'http://localhost:5173',
  credentials: true
}));
```

### Environment Variables

**Express (.env):**
```
MONGODB_URI=mongodb+srv://...
JWT_SECRET=<random-secret>
PORT=5001
BRAINTREE_MERCHANT_ID=...
BRAINTREE_PUBLIC_KEY=...
BRAINTREE_PRIVATE_KEY=...
RAZORPAY_KEY_ID=...
RAZORPAY_KEY_SECRET=...
```

**Python (.env):**
```
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=mistral-indian-law:latest
LLM_TEMPERATURE=0.1

# Server Configuration
PYTHON_PORT=8000
HOST=0.0.0.0

# External APIs
INDIAN_KANOON_API_KEY=...
LAWYER_API_KEY=...

# Performance Settings
MAX_DOCUMENT_SIZE_MB=10
CACHE_TTL_SECONDS=3600
```

### Payment Security

- **PCI Compliance:** Using Braintree/Razorpay Drop-in UI
- **No card storage:** All payment data handled by gateway
- **Webhook verification:** Validate payment callbacks

---

## Scalability & Performance

### Current Limitations

| Component | Limitation | Scale |
|-----------|-----------|-------|
| Express API | Single instance | ~1000 req/s |
| FastAPI | Synchronous agents | ~10 concurrent analyses |
| ChromaDB | Local file storage | ~100K documents |
| MongoDB Atlas | Free/Shared tier | Limited connections |

### Optimization Strategies

#### 1. Backend Scaling

**Horizontal Scaling:**
- Deploy multiple Express instances behind load balancer
- Use PM2 for cluster mode
- Implement Redis for session/cache management

**Vertical Scaling:**
- Increase MongoDB Atlas tier
- Upgrade server resources

#### 2. AI Service Optimization

**Asynchronous Processing:**
- Using `asyncio.gather()` for parallel RAG and Indian Kanoon queries
- LLM invocation via `run_in_executor` for non-blocking calls
- Session-based conversation history management

**Model Optimization:**
- Cached LLM instances with `@lru_cache`
- Fast LLM for intent classification (reduced `num_predict`)
- Local Ollama inference (no API latency)

**Performance Features:**
- Keyword-based fast-path for common intents
- Parallel initialization of Indian Kanoon and Crime RAG
- Configurable timeouts (15s for classification, 45s for generation)
- Session history limited to 20 messages for memory efficiency

#### 3. Database Optimization

**Indexing:**
```javascript
// MongoDB indexes
Lawyer.createIndex({ specialty: 1, rating: -1 });
Lawyer.createIndex({ location: 1 });
User.createIndex({ email: 1 }, { unique: true });
Booking.createIndex({ userId: 1, appointmentDate: -1 });
```

**Query Optimization:**
- Use projection to limit returned fields
- Implement pagination for large result sets
- Use aggregation pipelines for complex queries

#### 4. Frontend Optimization

**Code Splitting:**
- Lazy load routes with React.lazy()
- Dynamic imports for heavy components

**Caching:**
- Cache lawyer directory data
- Implement service worker for offline support

**Bundle Optimization:**
- Tree shaking (Vite handles this)
- Minimize third-party dependencies

#### 5. FAISS Scaling

**Options:**
- Use FAISS IVF index for larger document sets
- Implement index sharding for distributed search
- Consider GPU-accelerated FAISS for high-throughput
- Migrate to managed vector DB (Pinecone, Weaviate) for production scale

#### 6. Ollama Scaling

**Options:**
- Deploy multiple Ollama instances behind load balancer
- Use GPU acceleration for faster inference
- Consider cloud LLM APIs for production (with local fallback)
- Implement request queuing for high concurrency

---

## Deployment Strategy

### Development Environment

```
Client:  http://localhost:5173  (Vite dev server)
Express: http://localhost:5001  (ts-node)
FastAPI: http://127.0.0.1:8000  (uvicorn)
MongoDB: MongoDB Atlas (cloud)
```

### Production Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     Load Balancer                         │
│                   (Nginx/AWS ALB)                         │
└──────────────────────────────────────────────────────────┘
                    │
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
┌──────────────────┐  ┌──────────────────┐
│  Frontend CDN    │  │  API Cluster     │
│  (Vercel/        │  │  (Docker +       │
│   Netlify)       │  │   Kubernetes)    │
└──────────────────┘  └──────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
          ┌──────────────────┐  ┌──────────────────┐
          │  AI Service      │  │  MongoDB Atlas   │
          │  (Docker)        │  │  (Managed)       │
          └──────────────────┘  └──────────────────┘
```

### Deployment Steps

#### 1. Frontend (Vercel/Netlify)

```bash
cd client
npm run build
# Deploy build/ directory
```

**Environment Variables:**
```
VITE_API_BASE_URL=https://api.yourdomain.com
```

#### 2. Express Backend (Docker)

**Dockerfile:**
```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 5001
CMD ["npm", "start"]
```

**Deploy:**
```bash
docker build -t law-api:latest .
docker run -p 5001:5001 --env-file .env law-api:latest
```

#### 3. FastAPI Service (Docker)

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

# Install Tesseract OCR for image processing
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Copy FAISS index and data files
COPY app/data /app/app/data

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Note:** Ollama must be accessible from the container (use network mode or external URL).

#### 4. Kubernetes (Optional)

**Manifest Structure:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: law-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: law-api
  template:
    metadata:
      labels:
        app: law-api
    spec:
      containers:
      - name: api
        image: law-api:latest
        ports:
        - containerPort: 5001
        env:
        - name: MONGODB_URI
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: uri
```

### CI/CD Pipeline

**GitHub Actions Example:**

```yaml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy-frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to Vercel
        run: vercel --prod
        env:
          VERCEL_TOKEN: ${{ secrets.VERCEL_TOKEN }}

  deploy-backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build and Push Docker
        run: |
          docker build -t ${{ secrets.DOCKER_REGISTRY }}/law-api:latest .
          docker push ${{ secrets.DOCKER_REGISTRY }}/law-api:latest
      - name: Deploy to Kubernetes
        run: kubectl apply -f k8s/
```

### Monitoring & Logging

**Recommended Tools:**
- **Application Monitoring:** New Relic, DataDog
- **Logging:** Winston (Express), Loguru (Python)
- **Error Tracking:** Sentry
- **API Monitoring:** Postman Monitor, Uptime Robot

**Implementation:**
```typescript
// Express logging
import winston from 'winston';

const logger = winston.createLogger({
  level: 'info',
  format: winston.format.json(),
  transports: [
    new winston.transports.File({ filename: 'error.log', level: 'error' }),
    new winston.transports.File({ filename: 'combined.log' })
  ]
});
```

---

## Future Enhancements

### Short-term (3-6 months)
1. **Real-time Chat:** Implement WebSocket for live lawyer consultation
2. ~~**Document Upload:** Allow users to upload case documents for AI analysis~~ ✅ **COMPLETED**
3. ~~**OCR Support:** Image-based document analysis~~ ✅ **COMPLETED**
4. ~~**Document Validation:** 3-layer statutory compliance validation pipeline~~ ✅ **COMPLETED**
5. **Advanced Search:** Implement Elasticsearch for lawyer search
6. **Mobile App:** React Native or Flutter mobile application
7. **Admin Dashboard:** Lawyer management and analytics
8. **Multi-language OCR:** Support for Hindi and regional language documents

### Medium-term (6-12 months)
1. **Video Consultation:** Integrate WebRTC for video calls
2. **Multi-language Support:** Support regional Indian languages
3. **Case Management System:** Track ongoing cases
4. **Document Generation:** Auto-generate legal drafts
5. **Payment Subscriptions:** Subscription-based lawyer access

### Long-term (12+ months)
1. **AI Legal Assistant v2:** Fine-tuned LLM specifically for Indian law
2. **Predictive Analytics:** Case outcome prediction based on historical data
3. **Blockchain Integration:** Immutable case records
4. **Lawyer Marketplace:** Bidding system for cases
5. **Legal Education Platform:** Online courses and certifications

---

## Conclusion

The Law Education Platform is a modern, scalable full-stack application that combines traditional web development with cutting-edge AI technology. The system leverages:

- **Modern Web Stack:** React + TypeScript + Express for robust frontend and backend
- **AI/ML Innovation:** LangGraph state machine with local Ollama LLM for fast, cost-effective inference
- **3-Layer Document Validation:** Deterministic classification → rule-based statutory checklist → LLM-powered defect analysis with safety-first design (never renders binding legal opinions)
- **Comprehensive Document Processing:** OCR support for images, PDF/DOCX extraction, and legal document analysis
- **Legal Data Integration:** Indian Kanoon API for real-time case law and FAISS RAG for crime reporting guidance
- **Secure Architecture:** JWT authentication, encrypted payments, secure data handling
- **Scalable Design:** Microservices-ready architecture with clear separation of concerns

The system is production-ready with clear paths for scaling, monitoring, and future feature additions. The modular design allows for independent scaling of components and easy integration of new services.

---

**Last Updated:** January 28, 2026  
**Version:** 3.0  
**Author:** Law Education Platform Team
