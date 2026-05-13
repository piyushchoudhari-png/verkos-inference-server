from typing import Any

from pydantic import BaseModel


class AdminConfigResponse(BaseModel):
    config: dict[str, Any]
