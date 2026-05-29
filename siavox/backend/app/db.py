from datetime import datetime, timezone
from typing import Generator
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from backend.app.config import settings

# Engine setup (connect_args for SQLite threading compatibility)
connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(settings.DATABASE_URL, connect_args=connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    
    interactions = relationship("Interaction", back_populates="user", cascade="all, delete-orphan")

class Interaction(Base):
    __tablename__ = "interactions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    audio_path = Column(String, nullable=True)
    raw_transcript = Column(Text, nullable=True)
    transformed_text = Column(Text, nullable=False)
    format = Column(String, nullable=False)  # "note" or "message"
    source_language = Column(String, nullable=True)
    target_language = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    
    user = relationship("User", back_populates="interactions")

def init_db() -> None:
    """Initialize the database tables."""
    Base.metadata.create_all(bind=engine)

def get_db() -> Generator:
    """FastAPI Dependency for database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
