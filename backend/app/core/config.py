from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve the .env file relative to this file's location (backend/app/core/config.py
# → backend/.env) so the path is correct regardless of the working directory
# from which uvicorn or pytest is launched.
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    app_name: str = "XinYu Backend"
    service_name: str = "xinyu-backend"
    app_version: str = "0.1.0"
    api_prefix: str = "/api/v1"
    environment: str = "development"
    debug: bool = True
    log_level: str = "INFO"
    database_url: str = "postgresql+psycopg://postgres:postgres@127.0.0.1:5432/xinyu"

    # Doubao (Volcengine Ark) LLM provider — optional; only required when the
    # real DoubaoLLMProvider is active.  Tests use FakeLLMProvider instead.
    doubao_api_key: str = ""
    doubao_model: str = ""
    doubao_base_url: str = "https://ark.volcengine.com/api/v3"

    # NLP model path — set XINYU_NLP_MODEL_PATH to a trained artifact directory
    # (e.g. nlp/artifacts/roberta-domain) to activate RealNLPService.
    # When empty, StubNLPService is used and the backend starts without torch.
    nlp_model_path: str = ""

    # JWT authentication
    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24 * 7  # 7 days

    model_config = SettingsConfigDict(
        env_prefix="XINYU_",
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
