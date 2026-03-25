import pytest
import pandas as pd
from fastapi.testclient import TestClient
from main import app

@pytest.fixture
def client():
    """Создает тестовый клиент FastAPI"""
    return TestClient(app)

@pytest.fixture
def sample_data():
    """Генерирует синтетические данные для тестов"""
    data = {
        "monthly_fee": [20.0, 50.0],
        "usage_hours": [10.5, 30.0],
        "support_requests": [1, 5],
        "account_age_months": [12, 2],
        "failed_payments": [0, 1],
        "region": ["america", "europe"],
        "device_type": ["mobile", "desktop"],
        "payment_method": ["card", "card"],
        "autopay_enabled": [1, 0],
        "churn": [0, 1]
    }
    return pd.DataFrame(data)