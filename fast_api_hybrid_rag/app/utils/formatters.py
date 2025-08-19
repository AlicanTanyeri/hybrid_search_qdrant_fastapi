import re
import html
from datetime import datetime

def format_message_time(timestamp) -> str:
    """Timestamp'i güzel formata çevir"""
    try:
        if isinstance(timestamp, str):
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        else:
            dt = timestamp
        
        now = datetime.now()
        diff = now - dt.replace(tzinfo=None)
        
        if diff.days == 0:
            return dt.strftime('%H:%M')
        elif diff.days == 1:
            return 'Dün ' + dt.strftime('%H:%M')
        elif diff.days < 7:
            return dt.strftime('%a %H:%M')
        else:
            return dt.strftime('%d/%m/%Y')
    except:
        return "Bilinmeyen"

def clean_html_content(content: str) -> str:
    """HTML içeriğini temizle"""
    # HTML karakterlerini decode et
    content = html.unescape(content)
    
    # HTML taglerini temizle
    content = re.sub(r'<[^>]+>', '', content)
    
    # Güvenli HTML escape
    content = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    
    # Satır sonlarını HTML br'ye çevir
    content = content.replace("\n", "<br>")
    
    # Çok uzun satırları kır
    content = re.sub(r'(\S{50})', r'\1<wbr>', content)
    
    return content

def get_session_title(messages: list) -> str:
    """Session'ın başlığını ilk kullanıcı mesajından oluştur"""
    if not messages:
        return "Yeni Sohbet"
    
    # İlk kullanıcı mesajını bul
    first_user_message = next(
        (msg["content"] for msg in messages if msg["role"] == "user"),
        "Yeni Sohbet"
    )
    
    # İlk iki kelimeyi al ve sonuna "..." ekle
    words = first_user_message.split()
    if len(words) <= 2:
        return first_user_message
    else:
        return " ".join(words[:2]) + "..."