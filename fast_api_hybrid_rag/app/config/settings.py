import os
from pathlib import Path

# Uygulama yapılandırması
class Config:
    # Database
    DB_PATH = "chatbot.db"
    
    # Gemini API
    GEMINI_API_KEY = "api"
    GEMINI_MODEL = 'gemini-1.5-flash'
    
    # Qdrant
    QDRANT_URL = "http://localhost:6333"
    COLLECTION_NAME = "hybrid_search"
    
    # Modeller
    SENTENCE_TRANSFORMER_MODEL = 'emrecan/bert-base-turkish-cased-mean-nli-stsb-tr'
    TFIDF_VECTORIZER_PATH = 'path'
    
    # Streamlit
    PAGE_TITLE = "AI Assistant Pro"
    PAGE_ICON = "🧠"
    LAYOUT = "wide"
    
    # Limits
    MAX_SESSIONS = 15
    MAX_HISTORY_MESSAGES = 6
    MAX_RAG_RESULTS = 10
    MAX_SOURCE_TEXT_LENGTH = 350