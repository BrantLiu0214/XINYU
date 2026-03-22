from fastapi import APIRouter, Depends

from backend.app.dependencies.services import AppContainer, get_container
from backend.app.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(
    container: AppContainer = Depends(get_container),
) -> HealthResponse:
    settings = container.settings
    return HealthResponse(
        status="ok",
        service=settings.service_name,
        environment=settings.environment,
        version=settings.app_version,
    )
