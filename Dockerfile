# Используем легковесный образ Python
FROM python:3.10-slim

# Устанавливаем рабочую директорию
WORKDIR /code

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем библиотеки (без кэша, чтобы образ был меньше)
RUN pip install --no-cache-dir -r requirements.txt

# Копируем всё содержимое проекта в папку /code
COPY . .

# Создаем папку для моделей и данных (на случай, если репозиторий не создаст)
RUN mkdir -p storage data

# Открываем порт 8000
EXPOSE 8000

# Запускаем приложение через модуль (флаг -m)
# Мы указываем app.main:app, потому что main.py лежит в папке app/
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]