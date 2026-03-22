from backend.app.core.config import get_settings
from backend.app.dependencies.services import build_container


def test_backend_foundation_settings_defaults() -> None:
    settings = get_settings()
    assert settings.service_name == "xinyu-backend"
    assert settings.api_prefix == "/api/v1"
    assert settings.app_version == "0.1.0"


def test_backend_foundation_container_uses_cached_settings() -> None:
    container = build_container()
    assert container.settings is get_settings()
