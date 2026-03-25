from datetime import datetime
from typing import Any, Dict

from pydantic import BaseModel


class HistoryRecord(BaseModel):
      id: int
      model_name: str
      model_type: str
      hyperparameters: Dict[str, Any]
      metrics: Dict[str, float]
      timestamp: datetime
      status: str