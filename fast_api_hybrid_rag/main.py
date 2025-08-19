from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime

from app.config.settings import Config
from app.model.chatbot import ChatbotModel
from app.model.chatbot import DatabaseManager as db_manager

app = FastAPI(title="AI Assistant Pro", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None
    rag_sources: Optional[List[Dict[str, Any]]] = []

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None  

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
    session_id: str
    timestamp: str

class SessionResponse(BaseModel):
    session_id: str
    messages: List[ChatMessage]

class SessionListResponse(BaseModel):
    sessions: List[str]

chatbot_instance = None

@app.on_event("startup")
async def startup_event():
    """Uygulama başlangıcında chatbot'u initialize et"""
    global chatbot_instance
    try:
        chatbot_instance = ChatbotModel(Config)
        print("Chatbot başarıyla yüklendi")
    except Exception as e:
        print(f"Chatbot initialization error: {e}")
        raise

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Ana chat endpoint'i"""
    if not chatbot_instance:
        raise HTTPException(status_code=500, detail="Chatbot henüz yüklenmedi")
    
    try:
        session_id = request.session_id or str(uuid.uuid4())
        chatbot_instance.add_message(session_id, "user", request.message)
        response_result = chatbot_instance.generate_response(request.message, session_id)
        if isinstance(response_result, tuple) and len(response_result) == 2:
            assistant_reply, sources = response_result
        else:
            assistant_reply = response_result
            sources = []
        
        if sources is None:
            sources = []
        
        chatbot_instance.add_message(session_id, "assistant", assistant_reply, sources)
        
        return ChatResponse(
            response=assistant_reply,
            sources=sources,
            session_id=session_id,
            timestamp=datetime.now().strftime("%H:%M")
        )
        
    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat hatası: {str(e)}")

@app.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session_messages(session_id: str):
    """Belirli bir session'ın mesajlarını getir"""
    if not chatbot_instance:
        raise HTTPException(status_code=500, detail="Chatbot henüz yüklenmedi")
    
    try:
        messages = chatbot_instance.get_session_messages(session_id)
        if messages is None:
            messages = []
        
        formatted_messages = []
        for msg in messages:
            formatted_messages.append(ChatMessage(
                role=msg.get("role", ""),
                content=msg.get("content", ""),
                timestamp=msg.get("timestamp", ""),
                rag_sources=msg.get("rag_sources", [])
            ))
        
        return SessionResponse(
            session_id=session_id,
            messages=formatted_messages
        )
        
    except Exception as e:
        print(f"Session messages error: {e}")
        raise HTTPException(status_code=500, detail=f"Session mesajları hatası: {str(e)}")

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Session'ı sil"""
    if not chatbot_instance:
        raise HTTPException(status_code=500, detail="Chatbot henüz yüklenmedi")
    
    try:
        if hasattr(chatbot_instance, 'delete_session'):
            chatbot_instance.delete_session(session_id)
        return {"message": f"Session {session_id} silindi"}
    except Exception as e:
        print(f"Delete session error: {e}")
        raise HTTPException(status_code=500, detail=f"Session silme hatası: {str(e)}")

@app.post("/sessions/new")
async def create_new_session():
    """Yeni session oluştur"""
    session_id = str(uuid.uuid4())
    return {"session_id": session_id, "message": "Yeni session oluşturuldu"}

@app.get("/health")
async def health_check():
    """Health check endpoint'i"""
    stats = db_manager.get_statistics() if db_manager else {}
    
    return {
        "status": "healthy",
        "chatbot_loaded": chatbot_instance is not None,
        "database_loaded": db_manager is not None,
        "timestamp": datetime.now().isoformat(),
        "statistics": stats
    }

@app.get("/")
async def root():
    """Ana endpoint - API bilgileri"""
    return {
        "message": "AI Assistant Pro API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "POST /chat - Mesaj gönder (session_id opsiyonel - otomatik atanır)",
            "sessions": "GET /sessions - Tüm session'ları listele",
            "session_messages": "GET /sessions/{session_id} - Session mesajlarını getir",
            "delete_session": "DELETE /sessions/{session_id} - Session'ı sil",
            "new_session": "POST /sessions/new - Yeni session oluştur",
            "health": "GET /health - Sistem durumu ve istatistikler"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1905, reload=True)