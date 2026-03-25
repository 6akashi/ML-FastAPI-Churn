import json
import logging
import os

from fastapi import HTTPException
import joblib


from models.HistoryRecord import HistoryRecord
from models.ModelPipeline import ModelPipeline

logger = logging.getLogger(__name__)

class StorageRepository:
      @staticmethod
      def save_churn_model(pipeline, metadata, filename:str = "churn_model_v1"):
            path = os.path.join("storage", f"{filename}.joblib")
            joblib.dump(pipeline, path)
            logger.info(f"Сохранение модели в {path}")
            print(f"DEBUG: Looking for model at {path}")
            path = os.path.join("storage", f"{filename}.json")
            

            with open(path, "w", encoding="utf-8") as f:
                  json.dump(metadata, f, indent=4, ensure_ascii=False)
            logger.info("Метаданные модели сохранены успешно")

      @staticmethod
      def load_churn_model(filename:str = "churn_model_v1"):
            path_model = os.path.join("storage", f"{filename}.joblib")
            path_json = os.path.join("storage", f"{filename}.json")
            print(f"DEBUG: Looking for model at {path_model}")
            if os.path.exists(path_model) and os.path.exists(path_json):
                  logger.info(f"Загрузка существующей модели из {path_model}")
                  pipeline = joblib.load(path_model)
                  
                  with open(path_json, "r", encoding="utf-8") as f:
                        model_data = json.load(f)
                  model = ModelPipeline(pipeline,
                                          model_data["model_type"],
                                          model_data["hyperparameters"],
                                          model_data["last_train_time"],
                                          model_data["status"],
                                          model_data["metrics"]
                                          )
                  
                        
                  return model, model_data
            else:
                  logger.warning(f"Файлы модели {filename} не найдены в хранилище")
                  raise HTTPException(status_code=400, detail="Files NOT found at the path above!")
            
      @staticmethod
      def log_training(record: HistoryRecord, filenmae: str = 'training_history'):
            path = os.path.join('storage', f"{filenmae}.json")
            history = []
            logger.info(f"Добавление записи в историю обучения: {path}")
            if os.path.exists(path):
                  with open(path, "r", encoding="utf-8") as f:
                        try:
                              history = json.load(f)
                        except json.JSONDecodeError:
                              history = []
                        
            history.append(record.dict(by_alias=True))

            with open(path, "w", encoding="utf-8") as f:
                  json.dump(history, f, indent=4, default=str)
            logger.debug(f"История обновлена, ID записи: {record.id}")

      @staticmethod
      def get_history(filename: str = "training_history"):
            path = os.path.join("storage", f"{filename}.json")
            if not os.path.exists(path):
                  return []
            
            with open(path, 'r', encoding='utf-8') as f:
                  return json.load(f)