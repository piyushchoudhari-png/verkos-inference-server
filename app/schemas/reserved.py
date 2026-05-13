from pydantic import BaseModel


class NotImplementedResponse(BaseModel):
    detail: str
