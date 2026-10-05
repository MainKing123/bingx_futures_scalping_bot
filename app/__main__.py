import uvicorn
from app.runtime_v6 import RuntimeV6Settings

if __name__ == "__main__":
    settings = RuntimeV6Settings()
    uvicorn.run("app.main:app", host=settings.host, port=settings.port)
