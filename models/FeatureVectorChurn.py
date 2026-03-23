from pydantic import BaseModel

class FeatureVectorChurn(BaseModel):
      monthly_fee: float
      usage_hours: float
      support_request: int
      account_age_months: int
      failed_payments: int
      region: str
      device_type: str
      payment_method: str
      autopay_enabled: int
