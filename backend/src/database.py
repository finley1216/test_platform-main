# -*- coding: utf-8 -*-
"""
PostgreSQL database connection and session management
"""
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from src.config import config

# Database URL from config
SQLALCHEMY_DATABASE_URL = config.DATABASE_URL

# Create engine
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,  # Verify connections before using them
    pool_recycle=3600,   # Recycle connections after 1 hour
    pool_size=20,        # Increase pool size to handle concurrent polling
    max_overflow=30      # Allow more overflow connections
)

# Ensure pgvector extension is created
def ensure_pgvector_extension():
    """Ensure pgvector extension is created in PostgreSQL"""
    try:
        with engine.connect() as conn:
            # Execute CREATE EXTENSION IF NOT EXISTS vector
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
            print("✓ pgvector extension ensured")
    except Exception as e:
        print(f"⚠️  Warning: Could not ensure pgvector extension: {e}")
        print("   Please ensure pgvector is installed in PostgreSQL:")
        print("   CREATE EXTENSION IF NOT EXISTS vector;")

# Call on module import (only if pgvector is available)
try:
    from pgvector.sqlalchemy import Vector
    ensure_pgvector_extension()
except ImportError:
    print("⚠️  pgvector not installed, skipping extension creation")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency for FastAPI
def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

