import sqlite3

conn = sqlite3.connect("logs/chatlogs.db")
c = conn.cursor()
c.execute("CREATE TABLE IF NOT EXISTS logs (user TEXT, query TEXT, response TEXT)")

def log(user, query, response):
    c.execute("INSERT INTO logs VALUES (?, ?, ?)", (user, query, response))
    conn.commit()
