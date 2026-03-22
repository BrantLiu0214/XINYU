from fastapi import APIRouter

from backend.app.api.v1.chat import router as chat_router
from backend.app.api.v1.dashboard import router as dashboard_router
from backend.app.api.v1.health import router as health_router
from backend.app.api.v1.sessions import router as sessions_router

router = APIRouter()
router.include_router(health_router)
router.include_router(sessions_router, prefix="/sessions")
router.include_router(chat_router, prefix="/chat")
router.include_router(dashboard_router, prefix="/dashboard")
