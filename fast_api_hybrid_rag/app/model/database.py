import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional

class DatabaseManager:
    """SQLite veritabanı yöneticisi"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """SQLite veritabanını başlat"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                tokens_used INTEGER,
                rag_sources TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def get_connection(self):
        """Database bağlantısı"""
        return sqlite3.connect(self.db_path)
    
    def add_message(self, session_id: str, role: str, content: str, rag_sources: Optional[List] = None):
        """Mesaj ekle"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO conversations (session_id, role, content, rag_sources)
            VALUES (?, ?, ?, ?)
        """, (session_id, role, content, json.dumps(rag_sources) if rag_sources else None))
        
        conn.commit()
        conn.close()
    
    def get_session_messages(self, session_id: str) -> List[Dict]:
        """Belirli session'ın mesajlarını getir"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT role, content, timestamp, rag_sources
            FROM conversations
            WHERE session_id = ?
            ORDER BY timestamp ASC
        """, (session_id,))
        
        messages = cursor.fetchall()
        conn.close()
        
        return [
            {
                "role": msg[0],
                "content": msg[1],
                "timestamp": msg[2],
                "rag_sources": json.loads(msg[3]) if msg[3] else None
            }
            for msg in messages
        ]
    
    def get_all_sessions(self, limit: int = 15) -> List[Dict]:
        """Tüm chat session'larını getir"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT DISTINCT session_id,
                   MIN(timestamp) as first_message,
                   MAX(timestamp) as last_message,
                   COUNT(*) as message_count
            FROM conversations
            GROUP BY session_id
            ORDER BY last_message DESC
            LIMIT ?
        """, (limit,))
        
        sessions = cursor.fetchall()
        conn.close()
        
        return [
            {
                "session_id": s[0],
                "first_message": s[1],
                "last_message": s[2],
                "message_count": s[3],
                "title": f"Sohbet {s[0][-8:]}" if len(s[0]) > 8 else s[0]
            }
            for s in sessions
        ]
    
    def get_statistics(self) -> Dict:
        """Sistem istatistikleri"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Toplam mesaj sayısı
        cursor.execute("SELECT COUNT(*) FROM conversations")
        total_messages = cursor.fetchone()[0]
        
        # Toplam session sayısı
        cursor.execute("SELECT COUNT(DISTINCT session_id) FROM conversations")
        total_sessions = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_messages": total_messages,
            "total_sessions": total_sessions
        }