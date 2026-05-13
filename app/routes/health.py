from __future__ import annotations

from fastapi import APIRouter, Request

from app.schemas.health import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    engine = getattr(request.app.state, "engine", None)
    ready = engine is not None
    model_name = request.app.state.settings.model.name if ready else None
    return HealthResponse(status="ok", model=model_name, ready=ready)
