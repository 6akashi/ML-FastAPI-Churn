from datetime import datetime

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from models.ModelPipeline import ModelPipeline
from models.TrainingConfigChurn import TrainingConfigChurn
from services.ChurnDatasetModule import ChurnDatasetModule
from storage.StorageRepository import StorageRepository

MODEL_MAP = {
      "logistic_regression": LogisticRegression,
      "random_forest": RandomForestClassifier
}

class TrainingService:
      def __init__(self, reository: StorageRepository):
            self.repo = reository

      def run_training_pipeline(self, raw_data, config: TrainingConfigChurn):
            data_loader = ChurnDatasetModule()
            data_loader.data = raw_data
            X_train, X_test, y_train, y_test, y, num_features, cat_features = data_loader.split_data()

            preprocessor = ColumnTransformer(
                  transformers=[
                        ('num', StandardScaler(), num_features),
                        # Убирвем линейную зависимрсть(мультиколлинеарность), удаляя один из столбцов
                        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_features)
                  ]
            )

            model_class = MODEL_MAP.get(config.model_type.lower())
            if not model_class:
                  raise ValueError(f"Model {config.model_type} doesn't support, list: {list(MODEL_MAP.keys())}")
            
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

            pipline.fit(X_train, y_train)
            last_train = datetime.now()
            
            y_pred = pipline.predict(X_test)
            metrics = {
                  'accuracy': round(accuracy_score(y_test, y_pred), 4),
                  'f1-score': round(f1_score(y_test, y_pred, pos_label='Yes' if 'Yes' in y.values else 1), 4)
            }
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