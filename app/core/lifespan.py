from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI

from app.storage.StorageRepository import StorageRepository

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # При старте пытаемся загрузить последнюю сохраненную модель
    repo = StorageRepository()
    try:
        model, _ = repo.load_churn_model("latest_model")
        app.state.model = model
        print("Model loaded from disk")
    except Exception:
        # Если модели нет, ставим None
        app.state.model = None
        print("No model found on disk, waiting for /train")
    
    yield
    # Тут можно описать логику при выключении
    logger.info("Lifespan: Остановка приложения...")