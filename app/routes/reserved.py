from __future__ import annotations

from fastapi import APIRouter

from app.schemas.reserved import NotImplementedResponse

router = APIRouter()


@router.post("/v1/models/load", response_model=NotImplementedResponse, status_code=501)
async def load_model() -> NotImplementedResponse:
    return NotImplementedResponse(detail="not implemented")


@router.delete("/v1/models/{model_id}", response_model=NotImplementedResponse, status_code=501)
async def unload_model(model_id: str) -> NotImplementedResponse:
    return NotImplementedResponse(detail="not implemented")


@router.post(
    "/v1/models/{model_id}/warmup", response_model=NotImplementedResponse, status_code=501
)
async def warmup_model(model_id: str) -> NotImplementedResponse:
    return NotImplementedResponse(detail="not implemented")
