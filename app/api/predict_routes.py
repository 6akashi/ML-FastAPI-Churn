import logging
from typing import List, Union

from fastapi import APIRouter, Depends, Request, HTTPException

from app.core.dependencies import get_prediction_service
from app.errors.ErrorResponse import ErrorResponse
from app.models.FeatureVectorChurn import FeatureVectorChurn
from app.models.PredictionResponseChurn import PredictResponseChurn
from app.services.PredictionService import PredictionService

router = APIRouter(tags=["Inference"])
logger = logging.getLogger(__name__)

@router.post("/predict",
            response_model=PredictResponseChurn,
            responses={
                400: {
                    "model": ErrorResponse,
                    "description": "Model isn't ready to work",
                    "content": {
                        "application/json": {
                            "example": {
                                "code": "MODEL_NOT_READY",
                                "message": "Model not trained. Call /model/train first",
                                "details": None
                            }
                        }
                    }
                },
                422: {
                    "model": ErrorResponse,
                    "description": "Input features errors",
                    "content": {
                        "application/json": {
                            "example": {
                                "code": "FEATURE_MISMATCH",
                                "message": "Передано неверное количество признаков",
                                "details": {"expected": 10, "received": 8}
                            }
                        }
                    }
                }
            }
)
def predict(request: Request, features: Union[FeatureVectorChurn, List[FeatureVectorChurn]],
            service: PredictionService = Depends(get_prediction_service)):
      logger.info("Received prediction request")
      if not hasattr(request.app.state, "model") or request.app.state.model is None:
            raise HTTPException(
                  status_code=404, 
                  detail="Model isn't loaded. Please, train model by /model/train"
            )
      
      if logging.error:
           logger.error(f"Inference failed")
      return service.predict(features)