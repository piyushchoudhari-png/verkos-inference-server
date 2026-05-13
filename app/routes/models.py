import time

from fastapi import APIRouter, Request

from app.schemas.models import ModelCard, ModelList

router = APIRouter()


@router.get("/v1/models", response_model=ModelList)
async def list_models(request: Request) -> ModelList:
    settings = request.app.state.settings
    return ModelList(
        data=[
            ModelCard(
                id=settings.model.name,
                created=int(time.time()),
            )
        ]
    )
