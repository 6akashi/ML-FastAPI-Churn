from fastapi import APIRouter

from app.services.ChurnDatasetModule import ChurnDatasetModule


router = APIRouter(prefix="/dataset", tags=["Dataset info"])

data_loader = ChurnDatasetModule()

@router.get("/dataset/info")
def get_dataset_info():
      data_loader.load_from_csv("data/churn_dataset.csv")
      return data_loader.get_info()

@router.get("/dataset/split-info")
def dataset_split_info():
      data_loader.load_from_csv("data/churn_dataset.csv")
      X_train, X_test, y_train, y_test, y, num_features, cat_features = data_loader.split_data()

      return {
            "X_train": X_train.shape,
            "X_test": X_test.shape,
            "y_train": y_train.value_counts().to_dict(),
            "y_test": y_test.value_counts().to_dict(),
            "y": y.value_counts().to_dict(),
            "num_features": num_features,
            "cat_features": cat_features
      }
