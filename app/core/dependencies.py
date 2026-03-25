from fastapi import Request

from app.services.PredictionService import PredictionService
from app.services.TrainingService import TrainingService
from app.storage.StorageRepository import StorageRepository


def get_training_service() -> TrainingService:
      repo = StorageRepository()
      return TrainingService(repo)

def get_prediction_service(request: Request):
    # Передаем app.state.model (там наш ModelPipeline с методом .pipeline)
    if not hasattr(request.app.state, "model") or request.app.state.model is None:
        return PredictionService(None)
    return PredictionService(request.app.state.model)