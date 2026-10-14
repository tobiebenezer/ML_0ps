"""Core settings. Session 1 (serving) + Sessions 4-5 (monitoring) + retraining loop."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Breast Cancer MLOps API"
    model_path: str = "ml/model.joblib"
    baseline_path: str = "ml/baseline.json"
    holdout_path: str = "data/holdout.npz"
    base_train_path: str = "data/base_train.npz"
    feedback_db_path: str = "data/feedback.db"
    mlflow_model_uri: str = ""  # e.g. models:/breast-cancer-classifier/1 (optional)
    model_name: str = "breast-cancer-classifier"
    n_features: int = 30

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    latency_window: int = 500
    retrain_min_new_rows: int = 50
    retrain_cron_hour: str = "0"  # nightly; APScheduler cron hour
    cost_per_hour_usd: float = 0.10


settings = Settings()
