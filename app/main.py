from datetime import datetime
import logging
import os


from fastapi import FastAPI
from starlette.exceptions import HTTPException as StarletteHTTPException
import pandas as pd



from app.api import model_routes
from app.api import predict_routes

from app.core.exceptions import register_exception_handlers
from app.core.lifespan import lifespan

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("churn-service")

app = FastAPI(
    title="Churn Prediction Service",
    version="1.0.0",
    lifespan=lifespan
)



register_exception_handlers(app)

# Подключаем роутеры из папки api
app.include_router(model_routes.router)
app.include_router(predict_routes.router)

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": hasattr(app.state, "model") and app.state.model is not None,
        "dataset_present": os.path.exists("data/churn_dataset.csv"),
        "uptime": "up"
    }