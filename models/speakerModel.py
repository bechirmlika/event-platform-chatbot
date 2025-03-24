from config.db import get_db_connection

def search_speakers(query):
    """Search for speakers by first name, last name, or company."""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    sql = """
    SELECT firstName, lastName, company 
    FROM components_user_speakers 
    WHERE firstName LIKE %s OR lastName LIKE %s OR company LIKE %s
    """
    cursor.execute(sql, (f"%{query}%", f"%{query}%", f"%{query}%"))
    speakers = cursor.fetchall()

    cursor.close()
    conn.close()
    return speakers
