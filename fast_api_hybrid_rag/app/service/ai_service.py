import google.generativeai as genai
from typing import List, Dict, Tuple, Any, Callable
import json

class AIService:
    """AI model servisi (Gemini) - Tool call desteği ile"""
    
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.model = None
        self._initialize()
    
    def _initialize(self):
        """Gemini API'yi yapılandır"""
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
        except Exception as e:
            print(f"Gemini API yapılandırma hatası: {e}")
            raise
    
    def generate_response_with_tools(self, user_message: str, history: List[Dict] = None, 
                                   available_tools: List[Dict] = None, 
                                   tool_handler: Callable = None) -> str:
        """Tool call desteği ile AI yanıtı oluştur"""
        try:
            if history is None:
                history = []
            
            if not available_tools or not tool_handler:
                return self.generate_response(user_message, history, "")
            
            # İlk olarak tool kullanımına gerek var mı kontrol et
            should_use_tool = self._should_use_tools(user_message, history, available_tools)
            
            if should_use_tool:
                # Tool call yap
                tool_results = self._execute_tool_calls(user_message, available_tools, tool_handler)
                
                # Tool sonuçlarını kullanarak final response oluştur
                if tool_results:
                    return self._generate_final_response(user_message, history, tool_results)
            
            # Tool kullanmadan normal response
            return self.generate_response(user_message, history, "")
            
        except Exception as e:
            print(f"Tool-based response hatası: {e}")
            # Fallback to normal response
            return self.generate_response(user_message, history, "")
    
    def _should_use_tools(self, user_message: str, history: List[Dict], available_tools: List[Dict]) -> bool:
        """Tool kullanımına gerek var mı kontrol et"""
        try:
            # Basit keyword-based kontrol (geliştirilmesi gereken kısım)
            knowledge_keywords = [
                "nedir", "nasıl", "ne", "kim", "neden", "nerede", "ne zaman", "hangi",
                "açıkla", "anlat", "bilgi", "detay", "özellik", "avantaj", "dezavantaj",
                "fark", "karşılaştır", "örnekler", "kullanım", "uygulama", "yöntem",
                "çeşit", "tür", "kategori", "sınıf", "tanım", "kavram"
            ]
            
            user_lower = user_message.lower()
            return any(keyword in user_lower for keyword in knowledge_keywords)
            
        except:
            return False
    
    def _execute_tool_calls(self, user_message: str, available_tools: List[Dict], 
                          tool_handler: Callable) -> List[Dict]:
        """Tool call'ları execute et"""
        tool_results = []
        
        try:
            # Şimdilik sadece RAG tool'unu destekliyoruz
            for tool in available_tools:
                if tool.get("name") == "hybrid_rag_search":
                    result = tool_handler("hybrid_rag_search", {
                        "query": user_message,
                        "limit": 5
                    })
                    
                    if result.get("success") and result.get("found_relevant_content"):
                        tool_results.append(result)
                        break  # İlk başarılı sonuçla yetiniyoruz
            
        except Exception as e:
            print(f"Tool execution hatası: {e}")
        
        return tool_results
    
    def _generate_final_response(self, user_message: str, history: List[Dict], 
                               tool_results: List[Dict]) -> str:
        """Tool sonuçlarını kullanarak final response oluştur"""
        try:
            # Tool sonuçlarından context oluştur
            rag_context = ""
            source_info = []
            
            for result in tool_results:
                if result.get("context"):
                    rag_context += result["context"] + "\n\n"
                
                if result.get("sources"):
                    for source in result["sources"]:
                        source_info.append({
                            "title": source.get("title", "Bilinmeyen"),
                            "score": source.get("score", 0.0)
                        })
            
            # History formatla
            history_text = ""
            for msg in history[-5:]:  # Son 5 mesaj
                role = "Kullanıcı" if msg["role"] == "user" else "Asistan"
                history_text += f"{role}: {msg['content']}\n"
            
            # Enhanced prompt
            prompt = f"""Sen bir yardımcı asistansın. Aşağıdaki kaynakları kullanarak soruyu yanıtla.

Konuşma Geçmişi:
{history_text}

İlgili Kaynaklar:
{rag_context}

Soru: {user_message}

Yanıt verirken:
1. Kaynaklarda bulunan bilgileri kullan
2. Doğru ve güncel bilgiler ver
3. Türkçe yanıt ver
4. Eğer kaynaklarda yeterli bilgi yoksa, genel bilgilerinle tamamla
5. Kaynakları referans göstermek için başlıkları kullan

Yanıt:"""

            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            print(f"Final response generation hatası: {e}")
            return self.generate_response(user_message, history, "")
    
    def generate_response(self, user_message: str, conversation_history: List[Dict] = None, 
                        rag_context: str = "") -> str:
        """Standard AI yanıtı oluştur (legacy method)"""
        try:
            if conversation_history is None:
                conversation_history = []
                
            # Konuşma geçmişini formatla
            history_text = ""
            for msg in conversation_history:
                role = "Kullanıcı" if msg["role"] == "user" else "Asistan"
                history_text += f"{role}: {msg['content']}\n"
            
            # Prompt oluştur
            if rag_context:
                prompt = f"""Sen bir yardımcı asistansın. Aşağıdaki bilgileri kullanarak soruyu yanıtla.

Konuşma Geçmişi:
{history_text}

İlgili Kaynaklar:
{rag_context}

Yeni Soru: {user_message}

Lütfen kaynakları referans alarak yararlı ve detaylı bir yanıt ver. Türkçe yanıt ver."""
            else:
                prompt = f"""Sen bir yardımcı asistansın. Aşağıdaki konuşma geçmişini kullanarak soruyu yanıtla.

Konuşma Geçmişi:
{history_text}

Yeni Soru: {user_message}

Lütfen yararlı ve detaylı bir yanıt ver. Türkçe yanıt ver."""
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"Üzgünüm, bir hata oluştu: {str(e)}"
    
    def is_available(self) -> bool:
        """AI service'in kullanılabilir olup olmadığını kontrol et"""
        return self.model is not None
    
    def test_connection(self) -> Dict[str, Any]:
        """AI service bağlantısını test et"""
        try:
            if not self.model:
                return {
                    "status": "error",
                    "message": "Model başlatılmadı"
                }
            
            test_response = self.model.generate_content("Test mesajı")
            
            return {
                "status": "success",
                "message": "AI service aktif",
                "model": self.model_name,
                "test_response_length": len(test_response.text) if test_response.text else 0
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Test hatası: {e}",
                "model": self.model_name
            }