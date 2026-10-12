"""App package initialization."""

from app.chatbot import LegalChatbot, get_chatbot
from app.config import Settings, get_settings

__all__ = [
    "get_chatbot",
    "LegalChatbot",
    "get_settings",
    "Settings",
]
