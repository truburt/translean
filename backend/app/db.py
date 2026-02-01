"""
Copyright © 2026 Vladimir Vaulin-Belskii. All rights reserved.

Database session and engine configuration for the API.
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

from .config import settings

engine = create_async_engine(settings.database_url, echo=False, future=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
Base = declarative_base()


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide a transactional scope for FastAPI dependencies."""
    async with SessionLocal() as session:
        yield session
