from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
import os

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
# Factory to get a new session
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

def get_db():
    """FastAPI dependency that provides a database session and auto-closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
