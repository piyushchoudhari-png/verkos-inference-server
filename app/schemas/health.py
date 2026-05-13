from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model: str | None
    ready: bool
