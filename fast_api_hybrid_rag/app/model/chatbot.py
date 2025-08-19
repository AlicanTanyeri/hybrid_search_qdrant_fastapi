from typing import List, Dict, Tuple, Any
from app.model.database import DatabaseManager
from app.service.rag_service import RAGService
from app.service.ai_service import AIService


class ChatbotModel:
    """Ana chatbot modeli - Tool call desteği ile"""
    
    def __init__(self, config):
        self.config = config
        self.db_manager = DatabaseManager(config.DB_PATH)
        self.rag_service = RAGService.initialize(config)
        self.ai_service = AIService(config.GEMINI_API_KEY, config.GEMINI_MODEL)
    
    def get_available_tools(self) -> list[Dict]:
        """Mevcut tool'ları getir"""
        tools = []
        
        if self.rag_service.is_available():
            tools.append(self.rag_service.get_tool_definition())
        
        return tools
    
    def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Tool call'ı handle et"""
        if tool_name == "hybrid_rag_search":
            if self.rag_service.is_available():
                query = arguments.get("query", "")
                limit = arguments.get("limit", 5)
                return self.rag_service.handle_tool_call(query, limit)
            else:
                return {
                    "success": False,
                    "error": "RAG service not available",
                    "found_relevant_content": False
                }
        
        return {
            "success": False,
            "error": f"Unknown tool: {tool_name}",
            "found_relevant_content": False
        }
    
    def generate_response(self, user_message: str, session_id: str) -> Tuple[str, list[Dict]]:
        """Chatbot yanıtı oluştur - Tool call aware"""
        sources = []
        
        try:
            # Chat history
            history = self.db_manager.get_session_messages(session_id)[-self.config.MAX_HISTORY_MESSAGES:]
            
            # Available tools
            available_tools = self.get_available_tools()
            
            # AI response with potential tool usage
            assistant_reply = self.ai_service.generate_response_with_tools(
                user_message=user_message,
                history=history,
                available_tools=available_tools,
                tool_handler=self.handle_tool_call
            )
            
            # Extract sources if RAG was used
            if hasattr(assistant_reply, 'tool_results'):
                for tool_result in assistant_reply.tool_results:
                    if tool_result.get('found_relevant_content') and tool_result.get('sources'):
                        sources.extend(tool_result['sources'])
            
            if not assistant_reply or not assistant_reply.strip():
                assistant_reply = "Üzgünüm, şu anda yanıt oluşturamıyorum. Lütfen tekrar deneyin."
            
            return assistant_reply, sources
            
        except Exception as e:
            print(f"AI yanıt hatası: {e}")
            
            # Fallback: Try without tools
            try:
                history = self.db_manager.get_session_messages(session_id)[-self.config.MAX_HISTORY_MESSAGES:]
                assistant_reply = self.ai_service.generate_response(user_message, history, "")
                return assistant_reply or "Teknik bir sorun yaşandı. Lütfen daha sonra tekrar deneyin.", []
            except:
                return "Teknik bir sorun yaşandı. Lütfen daha sonra tekrar deneyin.", []
    
    def generate_response_legacy(self, user_message: str, session_id: str) -> Tuple[str, list[Dict]]:
        """Eski RAG yaklaşımı (backward compatibility için)"""
        rag_context = ""
        sources = []
        
        # RAG işlemi
        if self.rag_service.is_available():
            try:
                # Embedding oluştur
                query_dense = self.rag_service.dense_model.encode(user_message)
                if hasattr(query_dense, 'tolist'):
                    query_dense = query_dense.tolist()
                elif hasattr(query_dense, '__iter__'):
                    query_dense = list(query_dense)
                
                # RAG search
                rag_results = self.rag_service.hybrid_search(
                    query_text=user_message,
                    query_dense=query_dense,
                    limit=self.config.MAX_RAG_RESULTS
                )
                
                # Sonuçları işle
                if rag_results:
                    rag_context, sources = self.rag_service.extract_context_and_sources(
                        rag_results, 
                        self.config.MAX_SOURCE_TEXT_LENGTH
                    )        
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("RAG service is not available.")
            
        # AI yanıtı oluştur
        try:
            history = self.db_manager.get_session_messages(session_id)[-self.config.MAX_HISTORY_MESSAGES:]
            assistant_reply = self.ai_service.generate_response(user_message, history, rag_context)
            
            if not assistant_reply:
                assistant_reply = "Üzgünüm, şu anda yanıt oluşturamıyorum. Lütfen tekrar deneyin."
            
            return assistant_reply, sources
            
        except Exception as e:
            print(f"AI yanıt hatası: {e}")
            return "Teknik bir sorun yaşandı. Lütfen daha sonra tekrar deneyin.", []
    
    def add_message(self, session_id: str, role: str, content: str, rag_sources: list[Dict] = None):
        """Mesaj ekle"""
        try:
            self.db_manager.add_message(session_id, role, content, rag_sources)
        except Exception as e:
            print(f"Mesaj kaydetme hatası: {e}")
    
    def get_session_messages(self, session_id: str) -> list[Dict]:
        """Session mesajlarını getir"""
        try:
            return self.db_manager.get_session_messages(session_id)
        except Exception as e:
            print(f"Mesaj alma hatası: {e}")
            return []
    
    def get_all_sessions(self) -> List[Dict]:  
        """Tüm session'ları getir"""
        try:
            return self.db_manager.get_all_sessions(self.config.MAX_SESSIONS)
        except Exception as e:
            print(f"Session'ları alma hatası: {e}")
            return []
    
    def get_statistics(self) -> Dict:
        """İstatistikleri getir"""
        try:
            return self.db_manager.get_statistics()
        except Exception as e:
            print(f"İstatistik alma hatası: {e}")
            return {}
    
    def is_rag_available(self) -> bool:
        """RAG sisteminin durumu"""
        return self.rag_service.is_available()
    
    def test_rag_search(self, query: str) -> Dict:
        """RAG arama testi"""
        if not self.rag_service.is_available():
            return {"error": "RAG sistemi kullanılamıyor"}
        
        try:
            result = self.rag_service.handle_tool_call(query, limit=3)
            return {
                "success": result["success"],
                "found_content": result.get("found_relevant_content", False),
                "sources_count": len(result.get("sources", [])),
                "max_score": result.get("max_score", 0.0),
                "query": query
            }
        except Exception as e:
            return {"error": f"Test hatası: {e}"}
    
    def test_rag_connection(self) -> Dict:
        """RAG bağlantısını test et"""
        try:
            if not self.rag_service.is_available():
                return {
                    "status": "unavailable",
                    "message": "RAG sistemi başlatılamadı",
                    "components": {
                        "client": self.rag_service.client is not None,
                        "dense_model": self.rag_service.dense_model is not None,
                        "tfidf": self.rag_service.tfidf is not None
                    }
                }
            
            # Test search
            test_result = self.test_rag_search("test query")
            
            return {
                "status": "available" if test_result.get("success", False) else "error",
                "message": "RAG sistemi aktif" if test_result.get("success", False) else "RAG test başarısız",
                "test_result": test_result,
                "components": {
                    "client": True,
                    "dense_model": True, 
                    "tfidf": self.rag_service.tfidf is not None
                }
            }
            
        except Exception as e:
            return {
                "status": "error", 
                "message": f"Test hatası: {e}",
                "components": {"client": False, "dense_model": False, "tfidf": False}
            }