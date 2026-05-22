"""OpenAI-shaped error envelope helpers.

Every error response from the gateway carries this exact shape so that
OpenAI-compatible clients can parse failures without special-casing us::

    {"error": {"message": str, "type": str, "code": str|null, "param": str|null}}

See architecture doc §A.6.
"""

from typing import Any

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse


class GatewayError(HTTPException):
    """An HTTPException that already carries the OpenAI-shaped error body.

    Use this from route handlers so the central ``http_exception_handler`` can
    pass the body through verbatim. ``detail`` is the envelope's ``error``
    object — never a raw string.
    """

    def __init__(
        self,
        status_code: int,
        message: str,
        error_type: str,
        code: str | None = None,
        param: str | None = None,
    ) -> None:
        super().__init__(
            status_code=status_code,
            detail={"message": message, "type": error_type, "code": code, "param": param},
        )


def envelope(message: str, error_type: str, code: str | None = None, param: str | None = None) -> dict[str, Any]:
    """Build a bare error envelope dict (no Response wrapper)."""
    return {"error": {"message": message, "type": error_type, "code": code, "param": param}}


def error_response(
    request: Request,
    status_code: int,
    message: str,
    error_type: str,
    code: str | None = None,
    param: str | None = None,
) -> JSONResponse:
    """Build a JSONResponse with the envelope body and the request's x-request-id header."""
    request_id = getattr(request.state, "request_id", None)
    headers = {"x-request-id": request_id} if request_id else {}
    return JSONResponse(
        status_code=status_code,
        content=envelope(message, error_type, code, param),
        headers=headers,
    )


# Convenience constructors for the most common cases. Keep these aligned with the
# status-code → error-type table in architecture doc §A.6.


def invalid_request(message: str, param: str | None = None) -> GatewayError:
    return GatewayError(status.HTTP_400_BAD_REQUEST, message, "invalid_request_error", param=param)


def authentication_error(message: str = "missing or invalid bearer key") -> GatewayError:
    return GatewayError(status.HTTP_401_UNAUTHORIZED, message, "authentication_error")


def model_not_found(model_id: str) -> GatewayError:
    return GatewayError(
        status.HTTP_404_NOT_FOUND,
        f"model {model_id!r} is not in the catalog",
        "model_not_found",
        code="model_not_found",
        param="model",
    )


def worker_error(message: str) -> GatewayError:
    return GatewayError(status.HTTP_502_BAD_GATEWAY, message, "worker_error")


def service_unavailable(message: str) -> GatewayError:
    return GatewayError(status.HTTP_503_SERVICE_UNAVAILABLE, message, "service_unavailable")


def timeout_error(message: str = "upstream worker timed out") -> GatewayError:
    return GatewayError(status.HTTP_504_GATEWAY_TIMEOUT, message, "timeout")
