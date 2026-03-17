"""
Configuration module for the legal chatbot.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Ollama configuration
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "mistral-indian-law:latest"
    llm_temperature: float = 0.1

    # Server configuration - use PYTHON_PORT to avoid conflict with Express PORT
    host: str = "0.0.0.0"
    python_port: int = 8000

    # Express server port
    port_express: int = 5001

    # Database configuration
    mongodb_uri: str = ""

    # Authentication
    jwt_secret: str = ""

    # Braintree Sandbox API Keys
    braintree_merchant_id: str = ""
    braintree_public_key: str = ""
    braintree_private_key: str = ""

    # Optional external APIs
    lawyer_api_key: str = ""
    indian_kanoon_api_key: str = ""

    # Python Chatbot API Configuration
    python_api_url: str = "http://127.0.0.1:8000"

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


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
