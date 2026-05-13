from fastapi import APIRouter, Request

from app.schemas.admin import AdminConfigResponse

router = APIRouter()


@router.get("/admin/config", response_model=AdminConfigResponse)
async def get_config(request: Request) -> AdminConfigResponse:
    return AdminConfigResponse(config=request.app.state.settings.model_dump())
