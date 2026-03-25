import os

from fastapi.testclient import TestClient
import pandas as pd


def test_full_ml_workflow(client: TestClient):
    """
    Полный цикл: Обучение -> Статус -> Предсказание
    Используем синтетические данные для повторяемости.
    """
    # 1. Готовим синтетические данные (Пункт 4)
    # Вместо реального файла создаем микро-датасет на 10 строк
    test_csv_path = "data/test_churn.csv"
    os.makedirs("data", exist_ok=True)
    
    df = pd.DataFrame({
        "monthly_fee": [20.0] * 5 + [70.0] * 5,
        "usage_hours": [10.0] * 10,
        "support_requests": [1] * 10,
        "account_age_months": [12] * 10,
        "failed_payments": [0] * 10,
        "region": ["america"] * 10,
        "device_type": ["mobile"] * 10,
        "payment_method": ["card"] * 10,
        "autopay_enabled": [1] * 10,
        "churn": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] # 50/50 split
    })
    df.to_csv(test_csv_path, index=False)

    # 2. Обучение (/model/train)
    # ВАЖНО: твой сервис внутри model_train должен уметь брать путь к файлу, 
    # либо мы просто тестируем эндпоинт, который читает фиксированный data/churn_dataset.csv
    train_payload = {
        "model_type": "logistic_regression",
        "hyperparameters": {"C": 0.1}
    }
    train_response = client.post("/model/train", json=train_payload)
    assert train_response.status_code == 200
    assert "metrics" in train_response.json()

    # 3. Получение статуса (/model/status)
    status_response = client.get("/model/status")
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "Trained"
    assert "model_type" in status_response.json()

    # 4. Предсказание (/predict)
    predict_payload = [{
        "monthly_fee": 100.0, # Явный кандидат на отток
        "usage_hours": 1.0,
        "support_requests": 10,
        "account_age_months": 1,
        "failed_payments": 5,
        "region": "america",
        "device_type": "mobile",
        "payment_method": "card",
        "autopay_enabled": 0
    }]
    predict_response = client.post("/predict", json=predict_payload)
    
    assert predict_response.status_code == 200
    res_json = predict_response.json()
    assert res_json["status"] == "succes"
    assert len(res_json["results"]) == 1
    assert "prediction" in res_json["results"][0]
    assert "probabilities" in res_json["results"][0]

    # Чистим за собой тестовый файл
    if os.path.exists(test_csv_path):
        os.remove(test_csv_path)


def test_train_invalid_model_type(client: TestClient):
    payload = {
        "model_type": "super_ai_3000",
        "hyperparameters": {}
    }
    response = client.post("/model/train", json=payload)
    
    assert response.status_code == 400
    
    # Теперь проверяем ключ "message", который мы определили в ErrorResponse
    json_data = response.json()
    assert "message" in json_data
    assert "Model super_ai_3000 doesn't support..." in json_data["message"]
    assert json_data["code"] == "HTTP_ERROR"