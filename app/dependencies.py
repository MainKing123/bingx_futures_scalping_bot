from app.config import Settings
from app.db.engine import SessionLocal

settings = Settings()


async def get_db_session():
    async with SessionLocal() as session:
        yield session
