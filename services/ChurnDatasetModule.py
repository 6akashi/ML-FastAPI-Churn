from typing import List, Tuple

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


from models.DatasetRowChurn import DatasetRowChurn


class ChurnDatasetModule:
      def __init__(self):
            self.data: pd.DataFrame = pd.DataFrame()
            self.objects: List[DatasetRowChurn] = []

      def load_from_csv(self, file_path: str):
            # Read CSV file and save to DataFrame
            try:
                  self.data = pd.read_csv(file_path)
                  print(f"File {file_path} succesfully load. Strings: {len(self.data)}")
            except Exception as e:
                  print(f"Error, cannot read file: {e}")

      def transform_to_objects(self):
            # From DataFrame to DatasetRowChurn list
            if self.data.empty:
                  print("Data doesn't load")
                  return
            
            self.objects = [DatasetRowChurn(**row) for row in self.data.to_dict('records')]
            
      def get_info(self) -> dict:
            if self.data.empty:
                  return {"error": "Dataset empty or not loaded"}
            
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

            df = self._fill_popular_class(["monthly_fee", 
                                           "support_requests", 
                                           "failed_payments", 
                                           "region", 
                                           "device_type",
                                           "payment_method",
                                           "autopay_enabled"], df)

            df = self._fill_mean_class(["usage_hours", "account_age_months"], df)

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


      def train_churn_model(self):

            X_train, X_test, y_train, y_test, y, num_features, cat_features = self.split_data()

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

            y_pred = pipline.predict(X_test)
            metrics = {
                  'accuracy': round(accuracy_score(y_test, y_pred), 4),
                  'f1-score': round(f1_score(y_test, y_pred, pos_label='Yes' if 'Yes' in y.values else 1), 4)
            }
            return pipline, metrics


      def _fill_popular_class(self, list_classes: List[str], df: pd.DataFrame) -> pd.DataFrame:
            for cl in list_classes:
                  popular_cl = df[cl].value_counts().index[0]
                  df[cl] = df[cl].fillna(popular_cl)
            
            return df
      
      def _fill_mean_class(self, list_classes: List[str], df: pd.DataFrame) -> pd.DataFrame:
            for cl in list_classes:
                  mean = df[cl].mean()
                  df[cl] = df[cl].fillna(mean)

            return df


