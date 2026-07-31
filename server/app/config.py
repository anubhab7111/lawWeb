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

    # Where the bge-large embedding model runs: "auto" | "cuda" | "cpu".
    # Set EMBEDDINGS_DEVICE=cuda for one-off index rebuilds.
    embeddings_device: str = "auto"

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
