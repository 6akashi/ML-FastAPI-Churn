from datetime import datetime

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from models.ModelPipeline import ModelPipeline
from services.ChurnDatasetModule import ChurnDatasetModule
from storage.StorageRepository import StorageRepository


class TrainingService:
      def __init__(self, reository: StorageRepository):
            self.repo = reository

      def run_training_pipeline(self, raw_data):
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

            pipline = Pipeline(steps=[
                  ('preprocessor', preprocessor),
                  ('classifier', LogisticRegression(max_iter=1000))
            ])

            pipline.fit(X_train, y_train)
            last_train = datetime.now()
            
            y_pred = pipline.predict(X_test)
            metrics = {
                  'accuracy': round(accuracy_score(y_test, y_pred), 4),
                  'f1-score': round(f1_score(y_test, y_pred, pos_label='Yes' if 'Yes' in y.values else 1), 4)
            }
            model = ModelPipeline(pipline, last_train, "Trained", metrics)
            metadata = {
                "metrics": model.metrics,
                "last_train_time": model.time.isoformat(),
                "status": model.status
            }
            self.repo.save_churn_model(model.pipeline, metadata, "latest_model")

            return model