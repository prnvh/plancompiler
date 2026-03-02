import pandas as pd
import sqlite3

try:
    from sqlalchemy.engine import Engine as SAEngine
    _SQLALCHEMY_AVAILABLE = True
except ImportError:
    _SQLALCHEMY_AVAILABLE = False


def query_engine(conn, query: str) -> pd.DataFrame:
    """
    Executes SQL query on a DBHandle and returns a DataFrame.
    Accepts both sqlite3.Connection (from SQLiteConnector) and
    SQLAlchemy Engine (from PostgresConnector).
    Node: QueryEngine
    """
    is_sqlite = isinstance(conn, sqlite3.Connection)
    is_sqlalchemy = _SQLALCHEMY_AVAILABLE and isinstance(conn, SAEngine)

    if not is_sqlite and not is_sqlalchemy:
        raise TypeError(
            f"[QueryEngine] Unsupported DBHandle type: {type(conn).__name__}. "
            "Expected sqlite3.Connection or SQLAlchemy Engine."
        )

    df = pd.read_sql_query(query, conn)
    print("[QueryEngine] Query executed")
    return df