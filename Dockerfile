# 1. Базовый образ с Python
FROM python:3.10-slim

# 2. Установка рабочей директории
WORKDIR /app

# 3. Копируем зависимости и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Копируем весь код проекта
COPY . .

# 5. Открываем порт
EXPOSE 8000

# 6. Запуск сервера uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]