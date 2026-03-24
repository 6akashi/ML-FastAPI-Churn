import os

from fastapi import FastAPI

from models.FeatureVectorChurn import FeatureVectorChurn
from services.ChurnDatasetModule import ChurnDatasetModule

app = FastAPI()
dataset_loader = ChurnDatasetModule() 

@app.on_event("startup")
def startup_event():
      dataset_loader.load_churn_model("churn_model_v1.joblib")
      print("Startup: Model checked/loaded")

@app.get("/")
def say_hello():
      return {"message": "ml churn service is running"}

@app.post("/predict")
def predict(features: FeatureVectorChurn):
      return features

@app.get("/dataset/info")
def get_dataset_info():
      dataset_loader.load_from_csv("data/churn_dataset.csv")
      return dataset_loader.get_info()

@app.get("/dataset/split-info")
def dataset_split_info():
      dataset_loader.load_from_csv("data/churn_dataset.csv")
      return dataset_loader.split_data()

@app.post("/model/train")
def model_train():
      dataset_loader.load_from_csv("data/churn_dataset.csv")

      dataset_loader.train_churn_model()

      dataset_loader.save_churn_model("latest_model.joblib")
      
      return {
            "status": "Model trained succesfully",
            "metrics": dataset_loader.model.metrics
      }

@app.get("/model/status")
def get_model_status():
      if not dataset_loader.model:
            if not dataset_loader.load_churn_model("latest_model.joblib"):
                  return {"status": "Not trained", "message": "No model found"}
    
      return {
          "status": dataset_loader.model.status,
          "last_train_time": dataset_loader.model.time,
          "metrics": dataset_loader.model.metrics
      }


