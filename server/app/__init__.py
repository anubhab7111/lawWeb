"""
App package initialization.
"""

from app.chatbot import get_chatbot, LegalChatbot
from app.config import get_settings, Settings

__all__ = [
    "get_chatbot",
    "LegalChatbot",
    "get_settings",
    "Settings",
]
