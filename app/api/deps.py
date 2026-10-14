"""Shared FastAPI dependencies."""
from app.services import model_service


def get_model():
    return model_service.load_model()
