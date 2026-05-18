import base64
import io
import time
from typing import TYPE_CHECKING

from openai import AsyncOpenAI
from pydantic import BaseModel

if TYPE_CHECKING:
    import PIL.Image

    from src.config import PromptConfig, SimConfig


class InferResult(BaseModel):
    output_text: str | None
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float | None
    error_msg: str | None
    status: str  # "ok" | "error"


def _encode_image(image: "PIL.Image.Image") -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


async def infer_frame_openrouter(
    image: "PIL.Image.Image",
    prompt: "PromptConfig",
    frame_index: int,
    config: "SimConfig",
) -> InferResult:
    assert config.openrouter is not None
    or_cfg = config.openrouter

    client = AsyncOpenAI(
        api_key=or_cfg.api_key,
        base_url=or_cfg.base_url,
        timeout=or_cfg.timeout_s,
    )

    b64 = _encode_image(image)
    img_w: int = getattr(image, "width", 0)
    img_h: int = getattr(image, "height", 0)
    user_text = prompt.user.format(frame_index=frame_index, img_width=img_w, img_height=img_h)

    messages = [
        {"role": "system", "content": prompt.system},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                },
                {"type": "text", "text": user_text},
            ],
        },
    ]

    t0 = time.monotonic()
    ttft_s: float | None = None
    output_text: str | None = None
    prompt_tokens = 0
    completion_tokens = 0

    try:
        stream = await client.chat.completions.create(
            model=or_cfg.model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            stream=True,
            stream_options={"include_usage": True},
        )

        chunks: list[str] = []
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                if ttft_s is None:
                    ttft_s = time.monotonic() - t0
                chunks.append(chunk.choices[0].delta.content)
            if chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens
                completion_tokens = chunk.usage.completion_tokens

        output_text = "".join(chunks) or None

        return InferResult(
            output_text=output_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_s=ttft_s,
            error_msg=None,
            status="ok",
        )

    except Exception as exc:
        return InferResult(
            output_text=None,
            prompt_tokens=0,
            completion_tokens=0,
            ttft_s=None,
            error_msg=str(exc),
            status="error",
        )
