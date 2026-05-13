from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from pythonjsonlogger.jsonlogger import JsonFormatter

from app.config import load_settings
from app.engine import create_engine
from app.routes import admin, chat, health, metrics, models, reserved
from app.routes.metrics import set_model


def _configure_logging(log_level: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter(fmt="%(asctime)s %(levelname)s %(name)s %(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(log_level.upper())


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings, conflicts = load_settings()
    app.state.settings = settings
    app.state.engine = None

    _configure_logging(settings.server.log_level)
    logger = logging.getLogger("inference_server")

    for env_var, vals in conflicts.items():
        logger.warning(
            "Config override detected",
            extra={"env_var": env_var, "file_value": vals["file"], "env_value": vals["env"]},
        )

    logger.info(
        "Effective config",
        extra={
            "model_name": settings.model.name,
            "model_path": settings.model.path,
            "dtype": settings.model.dtype,
            "quantization": settings.model.quantization,
            "gpu_memory_utilization": settings.engine.gpu_memory_utilization,
            "tensor_parallel_size": settings.engine.tensor_parallel_size,
            "max_num_seqs": settings.model.max_num_seqs,
            "max_model_len": settings.model.max_model_len,
            **settings.engine.extra_kwargs(),
        },
    )

    app.state.engine = await create_engine(settings)
    if app.state.engine is not None:
        set_model(settings.model.name)
        logger.info("Engine ready", extra={"model": settings.model.name})
    else:
        logger.warning(
            "vLLM not available — inference routes will return 503",
            extra={"model": settings.model.name},
        )

    yield

    if app.state.engine is not None:
        set_model(None)
    logger.info("Shutting down")


app = FastAPI(title="verkos-inference-server", version="0.1.0", lifespan=lifespan)

app.include_router(chat.router)
app.include_router(models.router)
app.include_router(health.router)
app.include_router(metrics.router)
app.include_router(admin.router)
app.include_router(reserved.router)
