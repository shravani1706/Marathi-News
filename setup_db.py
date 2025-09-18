# setup_db.py
import sqlite3

def insert_into_db(df, db_name="news_articles.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # Create table if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS news_articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            headline TEXT,
            label TEXT,
            clean_text TEXT
        )
    """)

    # Insert rows
    for _, row in df.iterrows():
        cursor.execute("INSERT INTO news_articles (headline, label) VALUES (?, ?)", 
                       (row["headline"], row["label"]))

    conn.commit()
    conn.close()
