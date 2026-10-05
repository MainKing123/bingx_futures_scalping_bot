from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
DATABASE_URL = "sqlite+aiosqlite:///./volium.db"
engine = create_async_engine(DATABASE_URL)
SessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
