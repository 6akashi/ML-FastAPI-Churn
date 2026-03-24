from typing import Dict, List

from pydantic import BaseModel


class SinglePrediction(BaseModel):
      prediction: int
      probabilities: Dict[str, float]

class PredictResponseChurn(BaseModel):
      status: str
      results: List[SinglePrediction]