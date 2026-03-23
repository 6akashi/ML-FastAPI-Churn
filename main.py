from fastapi import FastAPI

from models.FeatureVectorChurn import FeatureVectorChurn
from services.ChurnDatasetModule import ChurnDatasetModule

app = FastAPI()

@app.get("/")
def say_hello():
      return {"message": "ml churn service is running"}

@app.post("/predict")
def predict(features: FeatureVectorChurn):
      return features

@app.get("/dataset/info")
def get_dataset_info():
      dataset_loader = ChurnDatasetModule() 
      dataset_loader.load_from_csv("data/churn_dataset.csv")
      return dataset_loader.get_info()

@app.get("/dataset/split-info")
def dataset_split_info():
      dataset_loader = ChurnDatasetModule() 
      dataset_loader.load_from_csv("data/churn_dataset.csv")
      return dataset_loader.split_data()

      
