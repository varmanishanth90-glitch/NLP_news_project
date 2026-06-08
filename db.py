import os
from dotenv import load_dotenv
import mysql.connector

load_dotenv()


def _get_mysql_connection():
    host = os.getenv("MYSQL_HOST")
    user = os.getenv("MYSQL_USER")
    password = os.getenv("MYSQL_PASSWORD")
    database = os.getenv("MYSQL_DATABASE")

    missing = [name for name, value in {
        "MYSQL_HOST": host,
        "MYSQL_USER": user,
        "MYSQL_PASSWORD": password,
        "MYSQL_DATABASE": database,
    }.items() if not value]

    if missing:
        raise RuntimeError(
            f"Missing required MySQL environment variables: {', '.join(missing)}"
        )

    return mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database,
    )


def get_news():
    conn = _get_mysql_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT DISTINCT id, title, url, published_at, source_name FROM google_news ORDER BY published_at DESC LIMIT 50"
    )
    rows = cursor.fetchall()
    conn.close()
    return rows
