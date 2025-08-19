import pickle
import warnings
from typing import Optional, Dict, Tuple, Any
from qdrant_client import models
from qdrant_client.models import NamedSparseVector, NamedVector, SparseVector, Prefetch

# Suppress sklearn version warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

class RAGTool:
    """RAG Tool for function calling"""
    
    @staticmethod
    def get_tool_definition():
        """Tool definition for LLM function calling"""
        return {
            "name": "hybrid_rag_search",
            "description": "Hybrid RAG search using dense and sparse vectors to find relevant documents",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query text"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20
                    }
                },
                "required": ["query"]
            }
        }
    
    @staticmethod
    def execute_tool(query: str, limit: int = 5, rag_service=None) -> Dict[str, Any]:
        """Execute RAG tool call"""
        if not rag_service or not rag_service.is_available():
            return {
                "success": False,
                "error": "RAG service not available",
                "results": []
            }
        
        try:
            # Dense vector oluştur
            query_dense = rag_service.dense_model.encode([query])[0].tolist()
            
            # RAG response al
            rag_response = rag_service.get_rag_response(query, query_dense, limit)
            
            if rag_response["use_rag"]:
                return {
                    "success": True,
                    "found_relevant_content": True,
                    "context": rag_response["context"],
                    "sources": rag_response["sources"],
                    "max_score": rag_response["max_score"],
                    "total_sources": rag_response["total_sources"]
                }
            else:
                return {
                    "success": True,
                    "found_relevant_content": False,
                    "reason": rag_response["reason"],
                    "max_score": rag_response["max_score"]
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "found_relevant_content": False
            }

class RAGService:
    """RAG (Retrieval Augmented Generation) servisi"""
    
    def __init__(self, qdrant_client=None, dense_model=None, tfidf_vectorizer=None, collection_name="hybrid_search"):
        self.client = qdrant_client
        self.dense_model = dense_model
        self.tfidf = tfidf_vectorizer
        self.collection_name = collection_name
        self.score_threshold = 0.30
    
    @classmethod
    def initialize(cls, config):
        """RAG servisini başlat"""
        try:
            from qdrant_client import QdrantClient
            from sentence_transformers import SentenceTransformer
            
            client = QdrantClient(url=config.QDRANT_URL)
            
            # Collection varlığını kontrol et
            collections = client.get_collections()
            collection_exists = any(col.name == config.COLLECTION_NAME for col in collections.collections)
            
            if not collection_exists:
                print(f"Collection '{config.COLLECTION_NAME}' bulunamadı!")
                return cls()
            
            dense_model = SentenceTransformer(config.SENTENCE_TRANSFORMER_MODEL, device="cpu")
            
            # TF-IDF yükleme
            tfidf = None
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with open(config.TFIDF_VECTORIZER_PATH, 'rb') as f:
                        tfidf = pickle.load(f)
                print("TF-IDF vectorizer yüklendi")
            except Exception as tfidf_error:
                print(f"TF-IDF yüklenemedi: {tfidf_error}")
            
            print("RAG sistemi başarıyla yüklendi!")
            return cls(client, dense_model, tfidf, config.COLLECTION_NAME)
            
        except Exception as e:
            print(f"RAG sistemi yüklenemedi: {e}")
            print("Sistem temel sohbet modunda çalışıyor")
            return cls()
    
    def is_available(self) -> bool:
        """RAG sisteminin kullanılabilir olup olmadığını kontrol et"""
        return all([self.client, self.dense_model])
    
    def get_tool_definition(self):
        """Get RAG tool definition for function calling"""
        return RAGTool.get_tool_definition()
    
    def handle_tool_call(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Handle RAG tool call"""
        return RAGTool.execute_tool(query, limit, self)
    
    def _create_sparse_vector(self, query_text: str) -> Optional[SparseVector]:
        """TF-IDF ile sparse vector oluştur"""
        if not self.tfidf or not query_text.strip():
            return None
        
        try:
            query_tfidf = self.tfidf.transform([query_text])
            if query_tfidf.nnz == 0:
                return None
                
            sparse_indices = query_tfidf[0].indices.tolist()
            sparse_values = query_tfidf[0].data.tolist()
            
            if sparse_indices and sparse_values:
                return SparseVector(indices=sparse_indices, values=sparse_values)
        except Exception as e:
            print(f"TF-IDF transform hatası: {e}")
        
        return None
    
    def hybrid_search(self, query_text: str, query_dense: list[float], limit: int = 10) -> Tuple[list, bool]:
        """TF-IDF ve dense vectorler ile hibrit arama"""
        if not query_text or not query_text.strip():
            return [], False
        
        if not query_dense or len(query_dense) == 0:
            return [], False
        
        limit = max(1, min(limit, 100))
        
        sparse_vector = self._create_sparse_vector(query_text)
        
        try:
            if sparse_vector:
                results = self.client.query_points(
                    collection_name=self.collection_name,
                    prefetch=[
                        models.Prefetch(
                            query=query_dense,
                            using="dense",
                            limit=limit,
                        ),
                        models.Prefetch(
                            query=sparse_vector,
                            using="sparse",
                            limit=limit * 2,
                        )
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=limit,
                    with_payload=True,
                    with_vectors=False
                )
            else:
                results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_dense,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False
                )
            
            points = self._process_search_results(results)
            
            if points:
                max_score = max(point.score for point in points if hasattr(point, 'score') and point.score is not None)
                use_rag = max_score >= self.score_threshold
                
                if use_rag:
                    filtered_points = []
                    for p in points:
                        if hasattr(p, 'score') and p.score >= self.score_threshold:
                            if hasattr(p, 'payload') and p.payload:
                                filtered_points.append(p)
                    
                    return filtered_points, True
                else:
                    return [], False
            else:
                return [], False
                
        except Exception as e:
            print(f"Search hatası: {e}")
            return [], False
    
    def _process_search_results(self, results) -> list:
        """Search sonuçlarını işle"""
        if not results:
            return []
        
        points = []
        
        if hasattr(results, 'points'):
            points = results.points or []
        elif isinstance(results, list):
            points = results
        else:
            return []
        
        if not points:
            return []
        
        valid_points = []
        
        for point in points:
            if not hasattr(point, 'id'):
                continue
            
            if not hasattr(point, 'payload') or point.payload is None:
                continue
            
            if not isinstance(point.payload, dict) or not point.payload:
                continue
            
            if hasattr(point, 'score') and point.score is not None:
                valid_points.append(point)
            else:
                point.score = 0.0
                valid_points.append(point)
        
        return valid_points
    
    def extract_context_and_sources(self, rag_results, max_text_length: int = 500) -> Tuple[str, list[Dict]]:
        """RAG sonuçlarından context ve kaynakları çıkar"""
        if not rag_results:
            return "", []
        
        rag_context = ""
        sources = []
        
        # rag_results tipini kontrol et
        if isinstance(rag_results, tuple):
            if len(rag_results) >= 1 and isinstance(rag_results[0], list):
                points_to_process = rag_results[0]
            else:
                return "", []
        elif isinstance(rag_results, list):
            points_to_process = rag_results
        elif hasattr(rag_results, 'points'):
            points_to_process = rag_results.points
        else:
            return "", []
        
        for i, point in enumerate(points_to_process):
            try:
                payload = point.payload if hasattr(point, 'payload') else None
                if not payload or not isinstance(payload, dict):
                    continue
                
                title = payload.get('original_title')
                text = payload.get('chunk_text') or "İçerik bulunamadı"
                chunk_index = payload.get('chunk_index', 0)
                total_chunks = payload.get('total_chunks', 1)
                original_doc_id = payload.get('original_doc_id') or 'unknown'
                
                text = str(text)
                if len(text) > max_text_length:
                    text = text[:max_text_length] + "..."
                
                score = float(point.score) if hasattr(point, 'score') and point.score else 0.0
                point_id = point.id if hasattr(point, 'id') else f'point_{i}'
                
                chunk_info = f" (Bölüm {chunk_index + 1}/{total_chunks})" if total_chunks > 1 else ""
                rag_context += f"**{title}{chunk_info}** (Score: {score:.3f})\n{text}\n\n"
                
                sources.append({
                    "point_id": str(point_id),
                    "original_doc_id": str(original_doc_id),
                    "title": title,
                    "chunk_index": chunk_index,
                    "total_chunks": total_chunks,
                    "score": round(score, 3),
                    "text_preview": text[:150] + "..." if len(text) > 150 else text
                })
                
            except Exception as e:
                print(f"Point {i} işleme hatası: {e}")
                continue
        
        return rag_context.strip(), sources
        
    def get_rag_response(self, query_text: str, query_dense: list[float], limit: int = 5) -> Dict:
        """RAG response'unu al"""
        results, use_rag = self.hybrid_search(query_text, query_dense, limit) 
        
        if not use_rag:
            return {
                "use_rag": False,
                "reason": f"En yüksek score {self.score_threshold} threshold'unun altında",
                "context": "",
                "sources": [],
                "max_score": 0.0
            }
        
        context, sources = self.extract_context_and_sources(results)
        max_score = max(point.score for point in results) if results else 0.0
        
        return {
            "use_rag": True,
            "context": context,
            "sources": sources,
            "max_score": round(max_score, 3),
            "total_sources": len(sources)
        }

# Tool kullanım örneği:
"""
# RAG service başlatma
rag_service = RAGService.initialize(config)

# Tool definition'ı alma
tool_def = rag_service.get_tool_definition()

# LLM'e tool'u tanıtma (OpenAI/Anthropic format)
tools = [tool_def]

# Tool call handling
def handle_function_call(function_name, arguments):
    if function_name == "hybrid_rag_search":
        query = arguments.get("query")
        limit = arguments.get("limit", 5)
        return rag_service.handle_tool_call(query, limit)
"""