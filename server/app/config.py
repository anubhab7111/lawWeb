"""Configuration module for the legal chatbot."""

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""


    ollama_base_url: str = "http://localhost:11434"


    llm_model: str = "qwen3:4b"

    fast_llm_model: str = "qwen3:4b"
    llm_temperature: float = 0.1


    llm_thinking: bool = True


    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_model_gpu: str = "BAAI/bge-reranker-v2-m3"


    reranker_device: str = "auto"


    rerank_blend: float = 0.5


    embeddings_device: str = "auto"


    ollama_vram_reserve_gb: float = 2.5


    corpus_root: str = Field(
        "", validation_alias=AliasChoices("LAWWEB_CORPUS_ROOT", "corpus_root")
    )


    embedding_model: str = "BAAI/bge-m3"
    embedding_query_instruction: str = ""


    currency: str = "USD"


    chat_thread_retention_days: int = 7

    session_ttl_seconds: int = 7200
    max_sessions: int = 500

    log_level: str = "INFO"


    clarify_on_ambiguous: bool = True


    route_tiebreak_enabled: bool = True
    route_tiebreak_timeout_seconds: float = 12.0


    retrieval_min_confidence: float = 0.45


    grounding_retry_enabled: bool = True
    grounding_retry_threshold: float = 0.5

    request_budget_seconds: int = 200


    concise_first_enabled: bool = True
    concise_first_max_query_words: int = 30


    llm_giveup_retry_enabled: bool = True
    llm_giveup_retry_max_elapsed_seconds: int = 330
    llm_retry_temperature: float = 0.4


    llm_breaker_failures: int = 3
    llm_breaker_cooldown_seconds: int = 30


    chat_rate_limit_per_minute: int = 20
    chat_max_concurrent: int = 4


    host: str = "0.0.0.0"
    python_port: int = 8000


    reload: bool = False


    debug: bool = False


    cors_allow_origins: str = (
        "http://localhost:3000,http://127.0.0.1:3000,"
        "http://localhost:5173,http://127.0.0.1:5173"
    )


    database_url: str = ""


    jwt_secret: str = ""


    braintree_merchant_id: str = ""
    braintree_public_key: str = ""
    braintree_private_key: str = ""


    lawyer_api_key: str = ""


    case_data_provider: str = "mock"
    case_data_api_key: str = ""
    case_data_api_base_url: str = ""


    google_client_id: str = ""
    google_client_secret: str = ""
    ms_client_id: str = ""
    ms_client_secret: str = ""
    public_api_url: str = "http://localhost:8000"
    client_app_url: str = "http://localhost:3000"
    token_enc_key: str = ""


    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_from_address: str = "no-reply@lawweb.local"
    fcm_service_account_json: str = ""


    firebase_web_config_json: str = ""
    firebase_vapid_key: str = ""


    r2_account_id: str = ""
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_bucket_name: str = "lawweb-vault"
    r2_endpoint_url: str = ""


    openrouter_api_key: str = ""

    openrouter_api_key_alt: str = ""


    openrouter_model: str = "nvidia/nemotron-3-super-120b-a12b:free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"


    openrouter_daily_limit: int = 50


    huggingface_token: str = ""


    max_document_size_mb: int = 10
    cache_ttl_seconds: int = 3600


    multilingual_enabled: bool = True
    language_detector: str = "fasttext"


    lang_detect_model_path: str = "app/data/models/lid.176.bin"


    lang_detect_min_confidence: float = 0.55
    default_language: str = "en"


    translation_model_indic_en: str = "adalat-ai/ct2-rotary-indictrans2-indic-en-dist-200M"
    translation_model_en_indic: str = "adalat-ai/ct2-rotary-indictrans2-en-indic-dist-200M"
    translation_device: str = "cpu"
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
