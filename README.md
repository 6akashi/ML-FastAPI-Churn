Сервис для обучения моделей и предсказания оттока клиентов на основе машинного обучения.

## 📊 Формат данных (churn_dataset.csv)
Датасет должен содержать следующие колонки:
- `monthly_fee`, `usage_hours`, `support_requests`, `account_age_months`, `failed_payments` (Числовые)
- `region`, `device_type`, `payment_method` (Категориальные)
- `churn` (Целевая переменная: 0 или 1)

## 🛠 Запуск
### Локально:
1. `pip install -r requirements.txt`
2. `uvicorn app.main:app --reload`

### Через Docker:
1. Сборка: `docker build -t churn-app .`
2. Запуск: `docker run -p 8000:8000 churn-app`

## 🔌 API Примеры
### Обучение модели
**POST** `/model/train`
```json
{
  "model_type": "random_forest",
  "hyperparameters": {"n_estimators": 100, "max_depth": 10}
}
```
Так же примеры есть в for_train.json

### Предсказания
**POST** `/predict`
```json
[{
  "monthly_fee": 50.5,
  "usage_hours": 120,
  "support_requests": 2,
  "account_age_months": 24,
  "failed_payments": 0,
  "region": "europe",
  "device_type": "mobile",
  "payment_method": "card",
  "autopay_enabled": 1
}]
```