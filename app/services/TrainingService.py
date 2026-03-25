from datetime import datetime
import logging

from fastapi import HTTPException
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from app.models.ModelPipeline import ModelPipeline
from app.models.TrainingConfigChurn import TrainingConfigChurn
from app.services.ChurnDatasetModule import ChurnDatasetModule
from app.storage.StorageRepository import StorageRepository

logger = logging.getLogger(__name__)

MODEL_MAP = {
      "logistic_regression": LogisticRegression,
      "random_forest": RandomForestClassifier
}

class TrainingService:
      def __init__(self, reository: StorageRepository):
            self.repo = reository

      def run_training_pipeline(self, raw_data, config: TrainingConfigChurn):
            logger.info(f"Запуск обучения модели типа: {config.model_type}")
            data_loader = ChurnDatasetModule()
            data_loader.data = raw_data
            X_train, X_test, y_train, y_test, y, num_features, cat_features = data_loader.split_data()

            logger.info(f"Данные разделены. Признаков: численных={len(num_features)}, категориальных={len(cat_features)}")

            numeric_transformer = Pipeline(steps=[
                  ('imputer', SimpleImputer(strategy="mean")),
                  ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                  ('imputer', SimpleImputer(strategy="most_frequent")),
                  ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
            ])

            preprocessor = ColumnTransformer(
                  transformers=[
                        ('num', numeric_transformer, num_features),
                        # Убирвем линейную зависимрсть(мультиколлинеарность), удаляя один из столбцов
                        ('cat', categorical_transformer, cat_features)
                  ]
            )

            model_class = MODEL_MAP.get(config.model_type.lower())
            if not model_class:
                  raise HTTPException(
                      status_code=400, 
                      detail=f"Model {config.model_type} doesn't support..."
                  )
            
            clean_params = {}
            for key, value in config.hyperparameters.items():
                  if isinstance(value, float) and value.is_integer():
                        clean_params[key] = int(value)
                  else:
                        clean_params[key] = value

            model_instance = model_class(**clean_params)


            pipline = Pipeline(steps=[
                  ('preprocessor', preprocessor),
                  ('classifier', model_instance)
            ])

            logger.info("Начало процесса fit (обучение)...")

            pipline.fit(X_train, y_train)
            last_train = datetime.now()
            
            y_pred = pipline.predict(X_test)
            metrics = {
                  'accuracy': round(accuracy_score(y_test, y_pred), 4),
                  'f1-score': round(f1_score(y_test, y_pred, pos_label='Yes' if 'Yes' in y.values else 1), 4)
            }

            logger.info(f"Обучение завершено. Метрики: {metrics}")
            
            model = ModelPipeline(pipline, config.model_type, config.hyperparameters, last_train, "Trained", metrics)
            metadata = {
                  "model_type": config.model_type,
                  "hyperparameters": config.hyperparameters,
                  "metrics": model.metrics,
                  "last_train_time": model.time.isoformat(),
                  "status": model.status
            }
            self.repo.save_churn_model(model.pipeline, metadata, "latest_model")

            return model