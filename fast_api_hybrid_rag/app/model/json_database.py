import json
from database import DatabaseManager  # DatabaseManager'ın olduğu dosya adı ile değiştir

def export_conversations_to_json(db_path: str, output_file: str):
    dbm = DatabaseManager(db_path)

    all_sessions = dbm.get_all_sessions()
    all_data = {}

    for session in all_sessions:
        session_id = session['session_id']
        messages = dbm.get_session_messages(session_id)
        all_data[session_id] = messages

    pretty_json = json.dumps(all_data, indent=4, ensure_ascii=False)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(pretty_json)
    print(f"Veriler '{output_file}' dosyasına başarıyla kaydedildi.")

if __name__ == "__main__":
    db_path = "/home/alicantanyeri/fast_api_hybrid_rag/chatbot.db"
    output_file = "all_conversations.json"
    export_conversations_to_json(db_path, output_file)
