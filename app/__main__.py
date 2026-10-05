import uvicorn
from app.runtime_settings import RuntimeSettings

if __name__ == "__main__":
    settings = RuntimeSettings()
    uvicorn.run("app.main:app", host=settings.host, port=settings.port)
