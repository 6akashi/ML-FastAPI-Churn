from datetime import datetime
import logging
import os
from typing import List, Optional, Union

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
import pandas as pd


from errors.ErrorResponse import ErrorResponse
from models.FeatureVectorChurn import FeatureVectorChurn
from models.HistoryRecord import HistoryRecord
from models.ModelPipeline import ModelPipeline
from models.PredictionResponseChurn import PredictResponseChurn
from models.TrainingConfigChurn import TrainingConfigChurn
from services.ChurnDatasetModule import ChurnDatasetModule
from services.PredictionService import PredictionService
from services.TrainingService import TrainingService
from storage.StorageRepository import StorageRepository

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("churn-service")

app = FastAPI()
data_loader = ChurnDatasetModule()
model_loader = StorageRepository() 
model: ModelPipeline

def get_training_service():
      repo = StorageRepository()
      return TrainingService(repo)

def get_prediction_service():
    # Передаем app.state.model (там наш ModelPipeline с методом .pipeline)
    if not hasattr(app.state, "model") or app.state.model is None:
        return PredictionService(None)
    return PredictionService(app.state.model)

@app.on_event("startup")
def startup_event():
      try:
            # Пытаемся загрузить модель
            repo = StorageRepository()
            model, metadata = repo.load_churn_model("latest_model")
            # Сохраняем прямо в объект приложения
            app.state.model = model
            app.state.model_status = "Loaded"
            print("Startup: Model loaded successfully")
      except Exception as e:
            app.state.model = None
            app.state.model_status = "Not Loaded"
            print(f"Startup: Model not found or error: {e}")

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
     return JSONResponse(
          status_code=exc.status_code,
          content=ErrorResponse(
               code="HTTP_ERROR",
               message=exc.detail
          ).dict()
     )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            code="VALIDATION_ERROR",
            message="Error at data structure",
            details=exc.errors()
        ).dict()
    )

@app.exception_handler(Exception)
async def common_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            code="INTERNAL_SERVER_ERROR",
            message="Произошла непредвиденная ошибка на сервере",
            details=str(exc) if os.getenv("DEBUG") else None
        ).dict()
    )

@app.get("/")
def say_hello():
      return {"message": "ml churn service is running"}

@app.post("/predict",
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
def predict(features: Union[FeatureVectorChurn, List[FeatureVectorChurn]],
            service: PredictionService = Depends(get_prediction_service)):
      logger.info("Received prediction request")
      if not hasattr(app.state, "model") or app.state.model is None:
            raise HTTPException(
                  status_code=404, 
                  detail="Model isn't loaded. Please, train model by /model/train"
            )
      
      if logging.error:
           logger.error(f"Inference failed")
      return service.predict(features)


@app.get("/dataset/info")
def get_dataset_info():
      data_loader.load_from_csv("data/churn_dataset.csv")
      return data_loader.get_info()

@app.get("/dataset/split-info")
def dataset_split_info():
      data_loader.load_from_csv("data/churn_dataset.csv")
      X_train, X_test, y_train, y_test, y, num_features, cat_features = data_loader.split_data()

      return {
            "X_train": X_train.shape,
            "X_test": X_test.shape,
            "y_train": y_train.value_counts().to_dict(),
            "y_test": y_test.value_counts().to_dict(),
            "y": y.value_counts().to_dict(),
            "num_features": num_features,
            "cat_features": cat_features
      }

@app.post("/model/train",
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
def model_train(config: TrainingConfigChurn, service: TrainingService = Depends(get_training_service)):
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

      app.state.model = model
      app.state.model_status = "Loaded"
      
      return {"status": "Trained and Logged", "id": new_id, "metrics": model.metrics}

@app.get("/model/status")
def get_model_status():
      if not hasattr(app.state, "model") or app.state.model is None:
        # Если в памяти нет, пробуем один раз загрузить с диска
        try:
            repo = StorageRepository()
            model, model_data = repo.load_churn_model("latest_model")
            app.state.model = model
        except Exception:
            return {"status": "Not Loaded", "message": "Model not found on disk or memory"}
            
    
      return {
        "model_type": app.state.model.model_type,
        "hyperparameters": app.state.model.hyperparameters,
        "status": app.state.model.status,
        "last_train_time": app.state.model.time,
        "metrics": app.state.model.metrics
    }

@app.get("/model/schema")
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

@app.get("/model/metrics")
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

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": hasattr(app.state, "model") and app.state.model is not None,
        "dataset_present": os.path.exists("data/churn_dataset.csv"),
        "uptime": "up"
    }