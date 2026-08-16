"""Lambda handler for the watershed extraction API.

Parses Function URL events (API Gateway v2 payload format), validates with
Pydantic, and passes to the pipeline. The processing logic is in pipeline.py;
this module only handles event parsing and error-to-HTTP conversion.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any

from pydantic import ValidationError

from watershed.models import PourPoint
from watershed.service.pipeline import run_pipeline
from watershed.service.schemas import (
    BBox,
    ErrorResponse,
    ExtractRequest,
    ExtractResponse,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _json_response(status_code: int, body: dict[str, Any]) -> dict[str, Any]:
    """Build a Lambda Function URL response dict."""
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def _error_response(status_code: int, error: str, message: str) -> dict[str, Any]:
    resp = ErrorResponse(error=error, message=message)
    return _json_response(status_code, resp.model_dump())


def _build_data_url(fmt: str, dataset: str | None) -> str:
    """Build the S3 URL for input flow direction data.

    Bucket is fixed via the INPUT_BUCKET_NAME environment variable. Key is
    either the request's dataset (benchmark variant) or the default from
    environment variables (COG_S3_KEY / ZARR_S3_KEY).
    """
    bucket = os.environ.get("INPUT_BUCKET_NAME", "")
    if not bucket:
        raise RuntimeError("INPUT_BUCKET_NAME environment variable is not set")

    if dataset is not None:
        key = dataset
    else:
        env_name = "COG_S3_KEY" if fmt == "cog" else "ZARR_S3_KEY"
        key = os.environ.get(env_name, "")
        if not key:
            raise RuntimeError(f"{env_name} environment variable is not set")
    return f"s3://{bucket}/{key}"


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Lambda entry point for the watershed extraction API."""
    # --- Route check ---
    method = event.get("requestContext", {}).get("http", {}).get("method", "")
    path = event.get("rawPath", "")
    if method != "POST" or path != "/extract-basin":
        return _error_response(
            400, "validation_error", f"Unsupported route: {method} {path}"
        )

    # --- Parse and validate request body ---
    raw_body = event.get("body", "")
    if not raw_body:
        return _error_response(400, "validation_error", "Request body is empty")

    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError:
        return _error_response(
            400, "validation_error", "Request body is not valid JSON"
        )

    try:
        req = ExtractRequest(**body)
    except ValidationError as exc:
        return _error_response(400, "validation_error", str(exc))

    # --- Extraction pipeline ---
    start = time.perf_counter()
    request_id = str(uuid.uuid4())

    try:
        data_url = _build_data_url(req.format, req.dataset)
        output = run_pipeline(
            data_url,
            req.format,
            PourPoint(latitude=req.latitude, longitude=req.longitude),
            request_id,
        )
    except ValueError as exc:
        # Distinguish non-flow cells (river mouth, sea, inland depression)
        message = str(exc)
        if "non-flow" in message.lower():
            return _error_response(400, "no_flow_data", message)
        return _error_response(400, "validation_error", message)
    except TimeoutError:
        return _error_response(504, "timeout", "Processing timed out")
    except Exception:
        logger.exception("Unexpected error during extraction")
        return _error_response(500, "internal_error", "Internal server error")

    logger.info("Per-phase timing: %s", output.phases)
    result = output.result

    resp = ExtractResponse(
        cell_count=result.cell_count,
        area_km2=result.area_km2,
        bbox=BBox(
            north=result.bbox.north,
            south=result.bbox.south,
            east=result.bbox.east,
            west=result.bbox.west,
        ),
        geotiff_url=output.geotiff_url,
        png_url=output.png_url,
        elapsed_sec=round(time.perf_counter() - start, 2),
        format=req.format,
        bfs_mode=result.bfs_mode,
        chunks_fetched=output.chunks_fetched,
        phases=output.phases,
    )
    return _json_response(200, resp.model_dump())
