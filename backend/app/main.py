from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.api.router import api_router
from backend.app.core.config import get_settings
from backend.app.core.logging import setup_logging

# Path: backend/app/main.py → project root is three levels up.
_FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(settings.log_level)

    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
    )

    # API routes registered first so they take priority over the SPA catch-all.
    application.include_router(api_router, prefix=settings.api_prefix)

    if _FRONTEND_DIST.exists():
        # Serve compiled JS/CSS assets from /assets/.
        application.mount(
            "/assets",
            StaticFiles(directory=_FRONTEND_DIST / "assets"),
            name="assets",
        )

        # Catch-all: serve root-level static files (favicon, etc.) as-is;
        # everything else returns index.html so React Router handles the route.
        @application.get("/{full_path:path}", include_in_schema=False)
        async def serve_spa(full_path: str) -> FileResponse:
            candidate = _FRONTEND_DIST / full_path
            if candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(_FRONTEND_DIST / "index.html")

    return application


app = create_app()
