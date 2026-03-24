import os
from typing import List, Union

from fastapi import Depends, FastAPI, HTTPException
import pandas as pd

from models.FeatureVectorChurn import FeatureVectorChurn
from models.ModelPipeline import ModelPipeline
from services.ChurnDatasetModule import ChurnDatasetModule
from services.PredictionService import PredictionService
from services.TrainingService import TrainingService
from storage.StorageRepository import StorageRepository

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

@app.get("/")
def say_hello():
      return {"message": "ml churn service is running"}

@app.post("/predict")
def predict(features: Union[FeatureVectorChurn, List[FeatureVectorChurn]],
            service: PredictionService = Depends(get_prediction_service)):
      
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

@app.post("/model/train")
def model_train(service: TrainingService = Depends(get_training_service)):
      data_loader.load_from_csv("data/churn_dataset.csv")
      model = service.run_training_pipeline(data_loader.data)
      
      app.state.model = model
      app.state.model_status = "Loaded"
      
      return {
            "status": "Model trained succesfully",
            "metrics": model.metrics
      }

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
        "status": app.state.model.status,
        "last_train_time": app.state.model.time,
        "metrics": app.state.model.metrics
    }

