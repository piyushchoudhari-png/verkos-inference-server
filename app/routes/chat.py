import base64
import json
import logging
import time
import uuid
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.routes.metrics import dec_in_flight, inc_in_flight, record_request
from app.schemas.chat import (
    ChatChoice,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Message,
    MessageOutput,
    StreamChoice,
    StreamDelta,
    Usage,
)

logger = logging.getLogger("inference_server.chat")

router = APIRouter()


def _extract_images(messages: list[Message]) -> list[Any]:
    try:
        import io

        from PIL import Image
    except ImportError:
        raise HTTPException(
            status_code=500, detail="Pillow not installed; required for image inputs"
        )

    images = []
    for msg in messages:
        if not isinstance(msg.content, list):
            continue
        for part in msg.content:
            if part.type == "image_url" and part.image_url:
                uri = part.image_url.url
                if not uri.startswith("data:"):
                    raise HTTPException(
                        status_code=400,
                        detail="Only base64 data URIs accepted for image inputs (no URL fetching in air-gapped mode)",
                    )
                _, encoded = uri.split(",", 1)
                raw = base64.b64decode(encoded)
                images.append(Image.open(io.BytesIO(raw)))
    return images


def _build_hf_messages(messages: list[Message]) -> list[dict[str, Any]]:
    result = []
    for msg in messages:
        if isinstance(msg.content, str):
            result.append({"role": msg.role, "content": msg.content})
        else:
            content: list[dict[str, Any]] = []
            for part in msg.content:
                if part.type == "image_url" and part.image_url:
                    content.append({"type": "image"})
                elif part.type == "text" and part.text:
                    content.append({"type": "text", "text": part.text})
            result.append({"role": msg.role, "content": content})
    return result


async def _generate(
    engine: Any,
    request: ChatCompletionRequest,
    model_name: str,
) -> AsyncIterator[tuple[Any, int]]:
    try:
        from vllm.sampling_params import SamplingParams
    except ImportError:
        raise HTTPException(status_code=503, detail="vLLM not available")

    request_id = str(uuid.uuid4())
    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        stop=request.stop,
        n=request.n,
    )

    images = _extract_images(request.messages)
    image_count = len(images)

    tokenizer = engine.get_tokenizer()
    prompt_text: str = tokenizer.apply_chat_template(
        _build_hf_messages(request.messages),
        tokenize=False,
        add_generation_prompt=True,
    )

    if images:
        inputs: Any = {
            "prompt": prompt_text,
            "multi_modal_data": {"image": images[0] if len(images) == 1 else images},
        }
    else:
        inputs = prompt_text

    async for output in engine.generate(inputs, sampling_params, request_id):
        yield output, image_count


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
) -> ChatCompletionResponse | StreamingResponse:
    engine = request.app.state.engine
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    settings = request.app.state.settings
    model_name = body.model or settings.model.name

    if body.stream:
        return StreamingResponse(
            _stream(engine, body, model_name, request),
            media_type="text/event-stream",
        )

    return await _complete(engine, body, model_name, request)


async def _complete(
    engine: Any,
    body: ChatCompletionRequest,
    model_name: str,
    request: Request,
) -> ChatCompletionResponse:
    t0 = time.monotonic()
    inc_in_flight()
    status = "success"
    usage: Usage | None = None

    try:
        final_output = None
        image_count = 0
        async for output, image_count in _generate(engine, body, model_name):
            final_output = output

        if final_output is None:
            raise HTTPException(status_code=500, detail="No output from engine")

        choices = [
            ChatChoice(
                index=i,
                message=MessageOutput(role="assistant", content=c.text),
                finish_reason=c.finish_reason,
            )
            for i, c in enumerate(final_output.outputs)
        ]
        usage = Usage(
            prompt_tokens=len(final_output.prompt_token_ids),
            completion_tokens=sum(len(c.token_ids) for c in final_output.outputs),
            total_tokens=len(final_output.prompt_token_ids)
            + sum(len(c.token_ids) for c in final_output.outputs),
        )
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=model_name,
            choices=choices,
            usage=usage,
        )
    except HTTPException:
        status = "error"
        raise
    except Exception as exc:
        status = "error"
        logger.exception("Inference error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        latency = time.monotonic() - t0
        dec_in_flight()
        record_request(status, latency)
        logger.info(
            "chat_completion",
            extra={
                "latency_ms": round(latency * 1000, 1),
                "prompt_tokens": usage.prompt_tokens if usage is not None else None,
                "completion_tokens": usage.completion_tokens
                if usage is not None
                else None,
                "image_count": image_count,
                "model": model_name,
                "status": status,
            },
        )

    return response


async def _stream(
    engine: Any,
    body: ChatCompletionRequest,
    model_name: str,
    request: Request,
) -> AsyncIterator[str]:
    t0 = time.monotonic()
    inc_in_flight()
    status = "success"
    image_count = 0
    completion_tokens = 0

    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    try:
        prev_texts: dict[int, str] = {}
        async for output, image_count in _generate(engine, body, model_name):
            for c in output.outputs:
                delta_text = c.text[len(prev_texts.get(c.index, "")) :]
                prev_texts[c.index] = c.text
                chunk = ChatCompletionChunk(
                    id=chunk_id,
                    created=created,
                    model=model_name,
                    choices=[
                        StreamChoice(
                            index=c.index,
                            delta=StreamDelta(content=delta_text),
                            finish_reason=c.finish_reason,
                        )
                    ],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
                if c.finish_reason:
                    completion_tokens += len(c.token_ids)
        yield "data: [DONE]\n\n"
    except Exception:
        status = "error"
        logger.exception("Streaming inference error")
        yield f"data: {json.dumps({'error': 'inference error'})}\n\n"
    finally:
        latency = time.monotonic() - t0
        dec_in_flight()
        record_request(status, latency)
        logger.info(
            "chat_completion_stream",
            extra={
                "latency_ms": round(latency * 1000, 1),
                "completion_tokens": completion_tokens,
                "image_count": image_count,
                "model": model_name,
                "status": status,
            },
        )
