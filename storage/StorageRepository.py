import json
import os

import joblib

from errors.Errors import LoadModelError
from models.ModelPipeline import ModelPipeline


class StorageRepository:
      @staticmethod
      def save_churn_model(pipeline, metadata, filename:str = "churn_model_v1"):
            path = os.path.join("storage", f"{filename}.joblib")
            joblib.dump(pipeline, path)
            
            path = os.path.join("storage", f"{filename}.json")
            with open(path, "w", encoding="utf-8") as f:
                  json.dump(metadata, f, indent=4, ensure_ascii=False)

      @staticmethod
      def load_churn_model(filename:str = "churn_model_v1"):
            path_model = os.path.join("storage", f"{filename}.joblib")
            path_json = os.path.join("storage", f"{filename}.json")
            if os.path.exists(path_model) and os.path.exists(path_json):
                  pipeline = joblib.load(path_model)
                  
                  with open(path_json, "r", encoding="utf-8") as f:
                        model_data = json.load(f)
                  model = ModelPipeline(pipeline,
                                          model_data["last_train_time"],
                                          model_data["status"],
                                          model_data["metrics"]
                                          )
                  
                        
                  return model, model_data
            else:
                  raise LoadModelError