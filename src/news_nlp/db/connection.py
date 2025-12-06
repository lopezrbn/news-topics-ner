from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
import os


def get_engine() -> Engine:
    """
    Create a SQLAlchemy Engine using database settings from environment variables.

    Expected environment variables:
      - DB_USER
      - DB_PASSWORD
      - DB_HOST
      - DB_PORT
      - DB_NAME
    """
    
    user = os.getenv("DB_USER", "news_nlp")
    password = os.getenv("DB_PASSWORD", "change_this_password")
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    dbname = os.getenv("DB_NAME", "news_nlp")

    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
    engine = create_engine(url)
    return engine
