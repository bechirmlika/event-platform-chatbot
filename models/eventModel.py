from config.db import get_db_connection

def search_events(query):
    """Search for events based on title, category, or description."""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    sql = """
    SELECT id, title, category, description, startDate, tags 
    FROM events 
    WHERE title LIKE %s OR category LIKE %s OR description LIKE %s OR tags LIKE %s
    """
    cursor.execute(sql, (f"%{query}%", f"%{query}%", f"%{query}%", f"%{query}%"))
    events = cursor.fetchall()

    cursor.close()
    conn.close()
    return events
