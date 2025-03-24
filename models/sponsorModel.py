from config.db import get_db_connection

def search_sponsors(query):
    """Search for sponsors by name or email."""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    sql = """
    SELECT name, email 
    FROM sponsors 
    WHERE name LIKE %s OR email LIKE %s
    """
    cursor.execute(sql, (f"%{query}%", f"%{query}%"))
    sponsors = cursor.fetchall()

    cursor.close()
    conn.close()
    return sponsors
