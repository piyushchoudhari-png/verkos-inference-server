from pydantic import BaseModel


class ImageURL(BaseModel):
    url: str  # must be a base64 data URI: "data:image/jpeg;base64,..."


class ContentPart(BaseModel):
    type: str  # "text" | "image_url"
    text: str | None = None
    image_url: ImageURL | None = None


class Message(BaseModel):
    role: str
    content: str | list[ContentPart]


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[Message]
    max_tokens: int | None = None
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: str | list[str] | None = None


class MessageOutput(BaseModel):
    role: str
    content: str


class ChatChoice(BaseModel):
    index: int
    message: MessageOutput
    finish_reason: str | None = None


class StreamDelta(BaseModel):
    role: str | None = None
    content: str | None = None


class StreamChoice(BaseModel):
    index: int
    delta: StreamDelta
    finish_reason: str | None = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: Usage


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamChoice]
