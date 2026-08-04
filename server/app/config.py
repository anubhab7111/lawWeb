"""
Configuration module for the legal chatbot.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Ollama configuration
    ollama_base_url: str = "http://localhost:11434"
    # llm_model: str = "mistral-indian-law:latest"
    llm_model: str = "qwen3:14b"
    # Small model for classification/routing/query-rewrite calls
    fast_llm_model: str = "qwen3:4b"
    llm_temperature: float = 0.1

    # Cross-encoder used to rerank fused BM25+dense candidates. Ranking
    # quality is what matters (scores are used relatively); the base model
    # keeps ~1.2GB of RAM free for the Ollama LLM on 16GB machines. Swap in
    # BAAI/bge-reranker-v2-m3 on larger hardware for a small quality bump.
    reranker_model: str = "BAAI/bge-reranker-base"

    # Where the reranker runs: "auto" | "cuda" | "cpu". Auto avoids small
    # (<6GB) GPUs entirely — the VRAM is worth more to Ollama's LLM offload.
    reranker_device: str = "auto"

    # Where the embedding model runs: "auto" | "cuda" | "cpu".
    # Set EMBEDDINGS_DEVICE=cuda for one-off index rebuilds.
    embeddings_device: str = "auto"

    # Shared dense embedding model. BGE-M3 is multilingual (100+ languages)
    # and still 1024-dim, so the pgvector columns and FAISS pipeline are
    # unchanged — but its indices are model-specific and must be rebuilt after
    # any change here. Unlike bge-large-en, M3 uses NO query-instruction
    # prefix; leaving embedding_query_instruction blank is required or
    # cross-lingual retrieval quality silently degrades.
    embedding_model: str = "BAAI/bge-m3"
    embedding_query_instruction: str = ""

    # Chat session lifecycle
    session_ttl_seconds: int = 7200
    max_sessions: int = 500

    # Server configuration
    host: str = "0.0.0.0"
    python_port: int = 8000

    # When True, unhandled errors return their message to the client (dev only).
    # Default False so 500 responses never leak internal exception details.
    debug: bool = False

    # Allowed CORS origins (comma-separated). Kept explicit rather than "*"
    # because allow_credentials=True + wildcard lets any origin make
    # credentialed requests. Default is the local Vite dev server.
    cors_allow_origins: str = "http://localhost:5173"

    # PostgreSQL connection string
    database_url: str = ""

    # Authentication
    jwt_secret: str = ""

    # Braintree Sandbox API Keys
    braintree_merchant_id: str = ""
    braintree_public_key: str = ""
    braintree_private_key: str = ""

    # Optional external APIs
    lawyer_api_key: str = ""
    indian_kanoon_api_key: str = ""

    # Case-data provider (My Cases / Hearing Reminders / Cause List Search).
    # "mock" (default) uses an in-memory fixture provider for local dev —
    # see app/tools/case_data_provider.py — until a licensed vendor
    # (e.g. eCourtsIndia) is contracted and its credentials set here.
    case_data_provider: str = "mock"
    case_data_api_key: str = ""
    case_data_api_base_url: str = ""

    # Notifications (Hearing Reminders / Smart Notifications). Left blank ->
    # notification_dispatch logs instead of sending (safe local-dev default).
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_from_address: str = "no-reply@lawweb.local"
    fcm_service_account_json: str = ""

    # Legal Document Vault object storage (Cloudflare R2, S3-compatible).
    # If unset, vault falls back to local disk under app/data/vault/ so the
    # feature is testable without live R2 credentials — see
    # app/services/object_storage.py.
    r2_account_id: str = ""
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_bucket_name: str = "lawweb-vault"
    r2_endpoint_url: str = ""

    # OpenRouter (LLM-as-judge for RAG evaluation — see app/metrics/llm_judge.py)
    openrouter_api_key: str = ""
    # Free-tier model; check https://openrouter.ai/models?max_price=0 for the
    # current catalog since free model availability rotates.
    openrouter_model: str = "openai/gpt-oss-20b:free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # OpenRouter free models: 20 req/min, 50 req/day (1000/day once the
    # account has $10+ in lifetime credit purchases). Bump via env var
    # after topping up rather than editing this default.
    openrouter_daily_limit: int = 50

    # Performance settings
    max_document_size_mb: int = 10
    cache_ttl_seconds: int = 3600

    # Multilingual support. Pipeline: detect language (fastText) → translate
    # query → English → run the existing English RAG/Qwen pipeline → translate
    # the English answer back to the user's language. Conversation memory stays
    # canonical-English. When disabled, the pipeline is a zero-overhead no-op
    # (no models load) and behaviour is identical to the English-only chatbot.
    multilingual_enabled: bool = True
    language_detector: str = "fasttext"
    # fastText language-id model (lid.176.bin, ~126MB). Path is resolved
    # relative to the server CWD, matching the data-path convention.
    lang_detect_model_path: str = "app/data/models/lid.176.bin"
    # Below this fastText confidence, assume the default language rather than
    # trust a shaky guess — short/code-mixed inputs are unreliable.
    lang_detect_min_confidence: float = 0.55
    default_language: str = "en"
    # IndicTrans2 distilled 200M checkpoints, one per direction. Distilled
    # keeps RAM ~0.8GB/direction (vs ~4GB for the 1B) — swap in the 1B via env
    # on larger hardware. Runs on CPU by default so the 4GB VRAM stays free for
    # Ollama's LLM offload.
    translation_model_indic_en: str = "ai4bharat/indictrans2-indic-en-dist-200M"
    translation_model_en_indic: str = "ai4bharat/indictrans2-en-indic-dist-200M"
    translation_device: str = "cpu"  # "auto" | "cuda" | "cpu"
    translation_cache: bool = True

    class Config:
        env_file = ".env"
        extra = "ignore"

    @property
    def port(self) -> int:
        """Return the Python server port."""
        return self.python_port

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse the comma-separated CORS origins into a list."""
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
