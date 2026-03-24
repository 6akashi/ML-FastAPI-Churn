from typing import List, Union

from fastapi import HTTPException
import pandas as pd

from models.FeatureVectorChurn import FeatureVectorChurn
from models.PredictionResponseChurn import PredictResponseChurn, SinglePrediction


class PredictionService:
      def __init__(self, model_container):
            self.model_container = model_container
      
      def predict(self, features: Union[FeatureVectorChurn, List[FeatureVectorChurn]]) -> PredictResponseChurn:
            if not self.model_container or not hasattr(self.model_container, "pipeline"):
                  raise HTTPException(
                      status_code=400,
                      detail="Model isn't trained. Call /model/train"
                  )
      
            data_list = [features] if isinstance(features, FeatureVectorChurn) else features
      
            input_df = pd.DataFrame([item.dict() for item in data_list])

            try:
           
                  expected_features = self.model_container.pipeline.feature_names_in_

                 
                  if input_df.shape[1] != len(expected_features):
                      raise HTTPException(
                          status_code=422,
                          detail={
                              "code": "FEATURE_COUNT_MISMATCH",
                              "message": f"Expected {len(expected_features)} features, but got {input_df.shape[1]}",
                              "expected": list(expected_features)
                          }
                      )

                  
                  input_df = input_df[expected_features]

            except AttributeError:
           
                  pass

            try:
                  preds = self.model_container.pipeline.predict(input_df)
                  probs = self.model_container.pipeline.predict_proba(input_df)

                  results = []

                  for i, p in enumerate(preds):
                        results.append(SinglePrediction(
                              prediction=int(p),
                              probabilities={
                                    "0": round(float(probs[i][0]), 4),
                                    "1": round(float(probs[i][1]), 4)
                              }
                        ))

                  return PredictResponseChurn(status="succes", results=results)
      
            except Exception as e:
                  raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")