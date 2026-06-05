import mysql.connector
from pprint import pprint

def get_news():
    conn = mysql.connector.connect(
        host="awsrdsdatabase.cwhsqi0qgwid.us-east-1.rds.amazonaws.com",
        user="admin",
        password="Eval4545",
        database="SQLRDSAWS"
    )
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, title, url, published_at, source_name FROM google_news ORDER BY published_at DESC LIMIT 50")
    rows = cursor.fetchall()
    conn.close()
    return rows
