from fastapi import FastAPI

from models.FeatureVectorChurn import FeatureVectorChurn

app = FastAPI()

@app.get("/")
def say_hello():
      return {"message": "ml churn service is running"}

@app.post("/predict")
def predict(features: FeatureVectorChurn):
      return features
