FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml ./
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" pydantic pydantic-settings joblib numpy scikit-learn scipy mlflow redis apscheduler python-multipart
COPY app/ app/
COPY ml/ ml/
COPY data/ data/
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
