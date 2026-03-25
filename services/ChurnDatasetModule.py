from datetime import datetime
import json
import logging
import os
from typing import List, Tuple

from fastapi import HTTPException
import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder



from models.DatasetRowChurn import DatasetRowChurn
from models.ModelPipeline import ModelPipeline

logger = logging.getLogger(__name__)

class ChurnDatasetModule:
      def __init__(self):
            self.data: pd.DataFrame = pd.DataFrame()
            self.objects: List[DatasetRowChurn] = []
            self.model = None

      def load_from_csv(self, file_path: str):
            # Read CSV file and save to DataFrame
            try:
                  logger.info(f"Load dataset from: {file_path}")
                  self.data = pd.read_csv(file_path)
                  if self.data.empty:
                      logger.warning(f"Файл {file_path} прочитан, но он пустой!")
                  else:
                      logger.info(f"Успешно загружено {len(self.data)} строк.")
            except Exception as e:
                  logger.error(f"Ошибка при чтении CSV {file_path}: {str(e)}")
                  raise HTTPException(status_code=400, detail="Dataset is empty or file isn't found. Train is impossible")

      def transform_to_objects(self):
            # From DataFrame to DatasetRowChurn list
            if self.data.empty:
                  raise HTTPException(status_code=400, detail="Data is not found")
            
            self.objects = [DatasetRowChurn(**row) for row in self.data.to_dict('records')]
            
      def get_info(self) -> dict:
            if self.data.empty:
                  raise HTTPException(status_code=400, detail="Dataset empty or not loaded")
            
            rows, columns = self.data.shape

            features = self.data.columns.tolist()

            churn_stats = self.data['churn'].value_counts().to_dict()
            
            x, y, numeric_features, categorial_features = self.prepare_data()

            return {
                  "rows_count": rows,
                  "columns_count": columns,
                  "feature_names": features,
                  "churn_dist": churn_stats,
                  'x': x.head().to_dict('records'),
                  "numeric": numeric_features,
                  "categorial": categorial_features
            }
      
      def prepare_data(self, target_column: str = 'churn') -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
            df: pd.DataFrame = self.data.copy()

            x: pd.DataFrame = df.drop(columns=[target_column])
            y: pd.Series = df[target_column]

            numeric_features = x.select_dtypes(include=["int64", "float64"]).columns.tolist()
            categorial_features = x.select_dtypes(include=["object", "category"]).columns.tolist()

            return x, y, numeric_features, categorial_features

      def split_data(self, test_size: float = 0.2, random_state: int = 42):

            x, y, num_features, cat_features = self.prepare_data()

            X_train, X_test, y_train, y_test = train_test_split(
                  x, y,
                  test_size=test_size,
                  random_state=random_state,
                  stratify=y
            )

            train_dist = {str(k): v for k, v in y_train.value_counts(normalize=True).to_dict().items()}
            test_dist = {str(k): v for k, v in y_test.value_counts(normalize=True).to_dict().items()}

            # return {
            #       "train_size": len(X_train),
            #       "test_size": len(X_test),
            #       "train_target_dist": train_dist,
            #       "test_target_dist": test_dist
            # }
            return X_train, X_test, y_train, y_test, y, num_features, cat_features



