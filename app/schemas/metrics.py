from pydantic import BaseModel


class GpuInfo(BaseModel):
    index: int
    memory_used_bytes: int
    memory_total_bytes: int


class MetricsResponse(BaseModel):
    uptime_seconds: float
    model: str | None
    requests: dict[str, int]
    in_flight: int
    avg_latency_seconds: float | None
    gpus: list[GpuInfo]
