"""
Configuration module for the legal chatbot.
"""

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Ollama configuration
    ollama_base_url: str = "http://localhost:11434"
    # llm_model: str = "mistral-indian-law:latest"
    # qwen3:14b Q4 (~9GB) spills off a 4GB GPU onto CPU (~3 tok/s, ~500s/answer).
    # qwen3:4b (~2.5GB) fits fully in VRAM and matches fast_llm_model, so Ollama
    # never swaps models mid-request.
    llm_model: str = "qwen3:4b"
    # Small model for classification/routing/query-rewrite calls
    fast_llm_model: str = "qwen3:4b"
    llm_temperature: float = 0.1
    # True for qwen3-style models whose output opens with an implicit thinking
    # block closed by </think>; False for models that answer directly.
    llm_thinking: bool = True

    # Cross-encoder used to rerank fused BM25+dense candidates (scores are used
    # relatively). On CPU the base model (~5s per 20 candidates); when the reranker
    # gets a GPU, the stronger multilingual v2-m3 (~3x the CPU cost of base).
    # No-pins retrieval eval, with rerank_blend 0.5: base hit@5 0.676 / MRR 0.539,
    # v2-m3 0.676 / 0.569 (the old pure-rerank base order: 0.559 / 0.464).
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_model_gpu: str = "BAAI/bge-reranker-v2-m3"

    # Where the reranker runs: "auto" | "cuda" | "cpu". Auto checks *live*
    # free VRAM against ollama_vram_reserve_gb (not a fixed card-size floor)
    # so it can use a small GPU when there's genuinely room, and falls back
    # to CPU when Ollama already holds most of the card.
    reranker_device: str = "auto"

    # Final order = weighted reciprocal-rank blend of the cross-encoder order and
    # the fused BM25+dense order (1.0 = cross-encoder only). Tuned on the
    # no-pins retrieval eval, where pure cross-encoder order lost recall.
    rerank_blend: float = 0.5

    # Where the embedding model runs: "auto" | "cuda" | "cpu".
    # Set EMBEDDINGS_DEVICE=cuda for one-off index rebuilds.
    embeddings_device: str = "auto"

    # VRAM to keep free for Ollama's LLM before "auto" embeddings/reranker
    # will claim the GPU. Default matches llm_model's footprint (qwen3:4b,
    # ~2.5GB) plus headroom for its KV cache; raise this if you configure a
    # bigger LLM. Checked live at call time against torch.cuda.mem_get_info(),
    # so it adapts to whatever Ollama is actually holding right now instead
    # of gating on total GPU capacity.
    ollama_vram_reserve_gb: float = 2.5

    # External drive holding raw/derived corpus data (see app/ingest/paths.py).
    # Empty = legacy layout with source data inside the repo's app/data.
    corpus_root: str = Field(
        "", validation_alias=AliasChoices("LAWWEB_CORPUS_ROOT", "corpus_root")
    )

    # Shared dense embedding model. BGE-M3 is multilingual (100+ languages)
    # and still 1024-dim, so the pgvector columns and FAISS pipeline are
    # unchanged — but its indices are model-specific and must be rebuilt after
    # any change here. Unlike bge-large-en, M3 uses NO query-instruction
    # prefix; leaving embedding_query_instruction blank is required or
    # cross-lingual retrieval quality silently degrades.
    embedding_model: str = "BAAI/bge-m3"
    embedding_query_instruction: str = ""

    # ISO code of the currency Braintree actually charges (sandbox: USD).
    currency: str = "USD"

    # Chat memory lives in Postgres checkpoints (app.checkpointing); a thread idle
    # this long is deleted by a scheduled job. Authenticated users lose nothing:
    # the next message re-seeds it from chat_messages.
    chat_thread_retention_days: int = 7
    # Only bound the in-process fallback used when Postgres is unavailable.
    session_ttl_seconds: int = 7200
    max_sessions: int = 500

    log_level: str = "INFO"

    # Agentic chat workflow (see app/chatbot.py). Ambiguous action-type queries
    # ("my landlord threatened me") get one clarifying question instead of a
    # silent guess between the law-explainer, crime-report and lawyer flows.
    clarify_on_ambiguous: bool = True
    # Cascade routing, tier 2 (see app/chatbot.py _resolve_ambiguity_with_llm):
    # before asking the clarifying question above, try one fast,
    # schema-constrained LLM call to resolve the near-tied route on its own.
    # Only reached on the minority of turns already about to interrupt the
    # user (clarify_on_ambiguous's own gate) — the common, unambiguous case
    # never pays this cost. Falls back to the clarifying question if the
    # model is also unsure, fails, or times out.
    route_tiebreak_enabled: bool = True
    route_tiebreak_timeout_seconds: float = 12.0
    # Retrieval grading: below this mean reranker score (or fewer than 3
    # provisions) the statute retrieval is treated as weak and retried once
    # with a rewritten, unfiltered query before generating. Set from a
    # 7-query sample (2026-09-21): in-corpus legal questions scored 0.58-1.00,
    # off-topic ones 0.34-0.40. Small sample — a false "weak" only costs one
    # extra retrieval pass, so re-tune against the eval set before relying on it.
    retrieval_min_confidence: float = 0.45
    # After generation, a grounding score below this triggers ONE regeneration
    # with retrieval targeted at the unsupported citations.
    grounding_retry_enabled: bool = True
    grounding_retry_threshold: float = 0.5
    # No retry/regeneration starts once a request has used this much wall time.
    request_budget_seconds: int = 200
    # Answer simple, well-retrieved questions with the concise prompt on the first
    # attempt (see _prefers_concise). "Simple" = retrieval graded good, a single
    # part, and no more than this many words; complex queries keep the full prompt.
    concise_first_enabled: bool = True
    concise_first_max_query_words: int = 30
    # When the model gives up (never closes its <think> block), retry once with
    # a trimmed, concise prompt at a higher temperature — but only if the request
    # is still younger than this. A give-up itself takes 2-3 minutes, so this is
    # deliberately much larger than request_budget_seconds.
    llm_giveup_retry_enabled: bool = True
    llm_giveup_retry_max_elapsed_seconds: int = 330
    llm_retry_temperature: float = 0.4
    # Ollama circuit breaker: after N consecutive failures/timeouts, fail fast
    # for the cooldown instead of making every request wait out the timeout.
    llm_breaker_failures: int = 3
    llm_breaker_cooldown_seconds: int = 30

    # Chat endpoint limits (in-memory, per process — the app runs one worker).
    chat_rate_limit_per_minute: int = 20
    chat_max_concurrent: int = 4

    # Server configuration
    host: str = "0.0.0.0"
    python_port: int = 8000
    # Auto-restart on source change. Off by default: every model/embedding/
    # reranker singleton and the RAG indices get reloaded from scratch on
    # each restart, which is exactly what NOT to trigger mid-demo. Set
    # RELOAD=true for local development.
    reload: bool = False

    # When True, unhandled errors return their message to the client (dev only).
    # Default False so 500 responses never leak internal exception details.
    debug: bool = False

    # Allowed CORS origins (comma-separated). Kept explicit rather than "*"
    # because allow_credentials=True + wildcard lets any origin make
    # credentialed requests. Covers both the client's configured dev port
    # (vite.config.ts: 3000) and Vite's own default (5173, e.g. docs/README
    # examples) so a fresh clone isn't CORS-blocked before anyone touches
    # .env — override for a LAN/deployed frontend origin.
    cors_allow_origins: str = (
        "http://localhost:3000,http://127.0.0.1:3000,"
        "http://localhost:5173,http://127.0.0.1:5173"
    )

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

    # Case-data provider (My Cases / Hearing Reminders / Cause List Search).
    # "mock" (default) uses an in-memory fixture provider for local dev —
    # see app/tools/case_data_provider.py — until a licensed vendor
    # (e.g. eCourtsIndia) is contracted and its credentials set here.
    case_data_provider: str = "mock"
    case_data_api_key: str = ""
    case_data_api_base_url: str = ""

    # Calendar sync (Google Calendar / Microsoft Outlook). A provider is offered
    # only when both its client id and secret are set. `public_api_url` is where
    # the browser reaches this server (OAuth redirect target); `client_app_url`
    # is where users land afterwards. `token_enc_key` encrypts stored refresh
    # tokens (falls back to a key derived from JWT_SECRET).
    google_client_id: str = ""
    google_client_secret: str = ""
    ms_client_id: str = ""
    ms_client_secret: str = ""
    public_api_url: str = "http://localhost:8000"
    client_app_url: str = "http://localhost:3000"
    token_enc_key: str = ""

    # Notifications (Hearing Reminders / Smart Notifications). Left blank ->
    # notification_dispatch logs instead of sending (safe local-dev default).
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_from_address: str = "no-reply@lawweb.local"
    fcm_service_account_json: str = ""
    # Public web-app config for browser push (Firebase console -> project settings),
    # as a JSON object string, plus the Web Push (VAPID) key pair's public key.
    firebase_web_config_json: str = ""
    firebase_vapid_key: str = ""

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
    # Second key, used once the first reaches openrouter_daily_limit or OpenRouter's quota.
    openrouter_api_key_alt: str = ""
    # Free-tier model; check https://openrouter.ai/models?max_price=0 for the
    # current catalog since free model availability rotates. openai/gpt-oss-20b:free
    # was retired (now 404s, paid-only) as of 2026-08-31, and minimax-m2.7:free
    # has since dropped off the free list too. nemotron-3-super-120b was checked
    # (2026-09-23) against the RAG-triad judge prompt: it parsed cleanly and
    # scored a grounded answer 1.0 vs an irrelevant one 0.0 on faithfulness.
    openrouter_model: str = "nvidia/nemotron-3-super-120b-a12b:free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # OpenRouter free models: 20 req/min, 50 req/day (1000/day once the
    # account has $10+ in lifetime credit purchases). Bump via env var
    # after topping up rather than editing this default.
    openrouter_daily_limit: int = 50

    # HuggingFace access token (IL-TUR benchmark dataset is gated — see
    # app/metrics/iltur_loader.py). Accept the license at
    # https://huggingface.co/datasets/Exploration-Lab/IL-TUR and generate a
    # token at https://huggingface.co/settings/tokens.
    huggingface_token: str = ""

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
    # IndicTrans2 distilled 200M checkpoints, one per direction, served via
    # CTranslate2. These are the non-gated CTranslate2 conversions of Raj Dabre's
    # rotary IndicTrans2 distilled models: they need no transformers modeling code
    # (which is incompatible with the transformers 5.x this stack runs on) and no
    # HuggingFace gating. Distilled keeps RAM ~0.5GB/direction. Runs on CPU by
    # default so the 4GB VRAM stays free for Ollama's LLM offload.
    translation_model_indic_en: str = "adalat-ai/ct2-rotary-indictrans2-indic-en-dist-200M"
    translation_model_en_indic: str = "adalat-ai/ct2-rotary-indictrans2-en-indic-dist-200M"
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
