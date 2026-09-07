from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

# Async engine talking to PostgreSQL via asyncpg
engine = create_async_engine(settings.DATABASE_URL, echo=False, future=True)

# Session factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Declarative base every ORM model inherits from."""
    pass


async def create_db_and_tables() -> None:
    """
    Inspects all models registered on Base.metadata and creates any missing
    tables/indexes directly in the database. Called once during app startup
    (see app.main.lifespan) before FastAPI accepts any HTTP requests.
    """
    # Import models here so they're registered on Base.metadata before create_all runs
    from app import models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding a DB session per request."""
    async with AsyncSessionLocal() as session:
        yield session
