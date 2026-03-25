from typing import Any, Dict

from pydantic import BaseModel


class TrainingConfigChurn(BaseModel):
      model_type: str
      hyperparameters: Dict[str, Any]