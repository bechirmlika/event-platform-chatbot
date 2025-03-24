from config.db import get_db_connection

def search_organizations(query):
    """Search for organizations by name or email."""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    sql = """
    SELECT libele, email 
    FROM organizations 
    WHERE libele LIKE %s OR email LIKE %s
    """
    cursor.execute(sql, (f"%{query}%", f"%{query}%"))
    organizations = cursor.fetchall()

    cursor.close()
    conn.close()
    return organizations
