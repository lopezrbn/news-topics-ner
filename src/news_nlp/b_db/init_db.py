"""
Initialize or update the PostgreSQL schema for the news_nlp project.

This script reads the SQL file at db/schema.sql and executes it against the
configured database. The schema file is written to be idempotent
(CREATE TABLE IF NOT EXISTS, CREATE INDEX IF NOT EXISTS), so it is safe
to run this script multiple times.
"""

from pathlib import Path
import sys
BASE_DIR = str(Path(__file__).resolve().parents[1])
if BASE_DIR not in sys.path:
    print(f"Adding {BASE_DIR} to sys.path")
    sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.engine import Engine

import a_config.paths as paths
from b_db.connection import get_engine


def read_schema_sql() -> str:
    """
    Read the schema.sql file from the db directory.

    Returns
    -------
    sql_text : str
        Full contents of the schema.sql file.
    """
    schema_path: Path = paths.SCHEMA_SQL_FILE
    if not schema_path.exists():
        raise FileNotFoundError(f"schema.sql not found at {schema_path}")

    return schema_path.read_text(encoding="utf-8")


def run_schema_sql(engine: Engine) -> None:
    """
    Execute the db/schema.sql file against the given database engine.

    The SQL file is split on ';' to execute each statement separately.
    This is sufficient for simple DDL statements like CREATE TABLE / INDEX.
    """
    sql_text = read_schema_sql()

    # Very simple split on ';'. We strip whitespace and ignore empty fragments.
    statements = [stmt.strip() for stmt in sql_text.split(";") if stmt.strip()]

    if not statements:
        print("No SQL statements found in schema.sql.")
        return

    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))

    print(f"Executed {len(statements)} SQL statements from schema.sql.")


def main() -> None:
    """
    Initialize or update the database schema by running db/schema.sql.

    This function:
      1) Loads environment variables from .env
      2) Creates a SQLAlchemy engine
      3) Executes the schema.sql script
    """
    # 1) Load environment variables from .env at the project root
    load_dotenv(paths.ENV_FILE)

    # 2) Create engine
    engine = get_engine()

    # 3) Run schema
    run_schema_sql(engine)

    print("Database schema initialized/updated successfully.")


if __name__ == "__main__":
    main()
