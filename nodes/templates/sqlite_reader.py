import sqlite3

def sqlite_reader(db_path: str):
    """
    Entry-point node that opens a pre-existing .db file from disk via db_path parameter and returns a DB handle. Use only to start a pipeline from a database that exists before the pipeline runs.
    Node: SQLiteReader
    """

    conn = sqlite3.connect(db_path)
    print(f"[SQLiteReader] Connected to {db_path}")
    return conn