from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Request


from app.core.dependencies import get_training_service
from app.errors.ErrorResponse import ErrorResponse
from app.models.HistoryRecord import HistoryRecord
from app.models.TrainingConfigChurn import TrainingConfigChurn
from app.services.ChurnDatasetModule import ChurnDatasetModule
from app.services.TrainingService import TrainingService
from app.storage.StorageRepository import StorageRepository


router = APIRouter(prefix="/model", tags=["Model Management"])

data_loader = ChurnDatasetModule()


@router.post("/train",
            responses={
                  400: {
                      "model": ErrorResponse,
                      "description": "Data error or invalid parametrizes",
                      "content": {
                          "application/json": {
                              "example": {
                                  "code": "TRAINING_FAILED",
                                  "message": "Dataset empty or not found",
                                  "details": "File data/churn_dataset.csv not found"
                              }
                          }
                      }
                  },
                  422: {
                      "model": ErrorResponse,
                      "description": "Data validation error (invalid model type)",
                      "content": {
                          "application/json": {
                              "example": {
                                  "code": "VALIDATION_ERROR",
                                  "message": "Unsupported model type",
                                  "details": "Value 'random_forest_v2' is not a valid model_type. Expected: ['logistic_regression', 'random_forest']"
                              }
                          }
                      }
                  }
            }     
    )
def model_train(request: Request, config: TrainingConfigChurn, service: TrainingService = Depends(get_training_service)):
      data_loader.load_from_csv("data/churn_dataset.csv")
      model = service.run_training_pipeline(data_loader.data, config)
      
      repo = StorageRepository()
      current_history = repo.get_history()
      new_id = len(current_history) + 1
      
      record = HistoryRecord(
        id=new_id,
        model_name=f"churn_v{new_id}",
        model_type=config.model_type,
        hyperparameters=config.hyperparameters,
        metrics=model.metrics,
        timestamp=datetime.now(),
        status="Trained"
    )
      repo.log_training(record)

      request.app.state.model = model
      request.app.state.model_status = "Loaded"
      
      return {"status": "Trained and Logged", "id": new_id, "metrics": model.metrics}

@router.get("/status")
def get_model_status(request: Request):
      if not hasattr(request.app.state, "model") or request.app.state.model is None:
        try:
            repo = StorageRepository()
            model, model_data = repo.load_churn_model("latest_model")
            request.app.state.model = model
        except Exception:
            return {"status": "Not Loaded", "message": "Model not found on disk or memory"}
            
    
      return {
        "model_type": request.app.state.model.model_type,
        "hyperparameters": request.app.state.model.hyperparameters,
        "status": request.app.state.model.status,
        "last_train_time": request.app.state.model.time,
        "metrics": request.app.state.model.metrics
    }

@router.get("/model/schema")
def get_feature_churn():
     return {
            "monthly_fee": "float",
            "usage_hours": "float",
            "support_requests": "int",
            "account_age_months": "int",
            "failed_payments": "int",
            "region": "str",
            "device_type": "str",
            "payment_method": "str",
            "autopay_enabled": "int",
     }

@router.get("/model/metrics")
def get_model_history(model_type: Optional[str] = None):
    repo = StorageRepository()
    history = repo.get_history()
    
    if model_type:
        history = [r for r in history if r['model_type'] == model_type]
    
    if not history:
        return {"message": "No history found"}


    history.reverse()
    
    return {
        "count": len(history),
        "last_train": history[0],
        "history": history[:10]
    }
