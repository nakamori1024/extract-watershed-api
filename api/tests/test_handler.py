"""Tests for handler (pipeline is mocked)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from watershed.models import BasinResult, BoundingBox, PourPoint
from watershed.service import handler
from watershed.service.pipeline import PipelineOutput


def make_event(
    body: str | None,
    method: str = "POST",
    path: str = "/extract-basin",
) -> dict[str, Any]:
    """Build a Function URL (API Gateway v2 format) event."""
    return {
        "requestContext": {"http": {"method": method}},
        "rawPath": path,
        "body": body,
    }


def make_pipeline_output() -> PipelineOutput:
    result = BasinResult(
        mask=np.ones((2, 2), dtype=np.uint8),
        bbox=BoundingBox(
            min_row=10,
            max_row=11,
            min_col=20,
            max_col=21,
            south=35.0,
            north=35.002,
            west=139.0,
            east=139.002,
        ),
        cell_count=4,
        area_km2=0.04,
        crs="EPSG:4326",
        transform=(0.001, 0.0, 139.0, 0.0, -0.001, 35.002),
        pour_point=PourPoint(35.001, 139.001),
        bfs_mode="chunked",
    )
    return PipelineOutput(
        result=result,
        geotiff_url="https://example/signed.tif",
        png_url="https://example/signed.png",
        chunks_fetched=3,
        phases={"reader_init": 0.5, "extract": 1.2},
    )


@pytest.fixture(autouse=True)
def input_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INPUT_BUCKET_NAME", "input-bucket")
    monkeypatch.setenv("COG_S3_KEY", "default/dir.tif")
    monkeypatch.setenv("ZARR_S3_KEY", "default/dir.zarr")


VALID_BODY = json.dumps({"latitude": 35.7446, "longitude": 140.8507, "format": "zarr"})


class TestRouting:
    def test_wrong_method_rejected(self) -> None:
        resp = handler.lambda_handler(make_event(VALID_BODY, method="GET"), None)
        assert resp["statusCode"] == 400

    def test_wrong_path_rejected(self) -> None:
        resp = handler.lambda_handler(make_event(VALID_BODY, path="/other"), None)
        assert resp["statusCode"] == 400


class TestValidation:
    def test_empty_body(self) -> None:
        resp = handler.lambda_handler(make_event(None), None)
        assert resp["statusCode"] == 400
        assert json.loads(resp["body"])["error"] == "validation_error"

    def test_invalid_json(self) -> None:
        resp = handler.lambda_handler(make_event("not json"), None)
        assert resp["statusCode"] == 400

    def test_out_of_bounds_latitude(self) -> None:
        body = json.dumps({"latitude": 50.0, "longitude": 140.0, "format": "cog"})
        resp = handler.lambda_handler(make_event(body), None)
        assert resp["statusCode"] == 400


class TestHappyPath:
    def test_returns_full_response(self) -> None:
        with patch.object(
            handler, "run_pipeline", return_value=make_pipeline_output()
        ) as mock_run:
            resp = handler.lambda_handler(make_event(VALID_BODY), None)

        assert resp["statusCode"] == 200
        body = json.loads(resp["body"])
        assert body["cell_count"] == 4
        assert body["bfs_mode"] == "chunked"
        assert body["chunks_fetched"] == 3
        assert body["phases"]["extract"] == 1.2
        assert body["geotiff_url"] == "https://example/signed.tif"
        # Default dataset (ZARR_S3_KEY from environment variable) is used
        assert mock_run.call_args.args[0] == "s3://input-bucket/default/dir.zarr"

    def test_dataset_overrides_default_key(self) -> None:
        body = json.dumps(
            {
                "latitude": 35.0,
                "longitude": 140.0,
                "format": "cog",
                "dataset": "jflwdir-bench/dir_cog_b256.tif",
            }
        )
        with patch.object(
            handler, "run_pipeline", return_value=make_pipeline_output()
        ) as mock_run:
            resp = handler.lambda_handler(make_event(body), None)

        assert resp["statusCode"] == 200
        assert (
            mock_run.call_args.args[0]
            == "s3://input-bucket/jflwdir-bench/dir_cog_b256.tif"
        )


class TestErrorMapping:
    def test_non_flow_cell_maps_to_no_flow_data(self) -> None:
        err = ValueError(
            "Coordinates (43.2, 141.3) fall on a non-flow cell (value=247)"
        )
        with patch.object(handler, "run_pipeline", side_effect=err):
            resp = handler.lambda_handler(make_event(VALID_BODY), None)
        assert resp["statusCode"] == 400
        assert json.loads(resp["body"])["error"] == "no_flow_data"

    def test_other_value_error_maps_to_validation_error(self) -> None:
        err = ValueError("Coordinates are outside the data extent")
        with patch.object(handler, "run_pipeline", side_effect=err):
            resp = handler.lambda_handler(make_event(VALID_BODY), None)
        assert resp["statusCode"] == 400
        assert json.loads(resp["body"])["error"] == "validation_error"

    def test_unexpected_error_maps_to_500(self) -> None:
        with patch.object(handler, "run_pipeline", side_effect=RuntimeError("boom")):
            resp = handler.lambda_handler(make_event(VALID_BODY), None)
        assert resp["statusCode"] == 500
        assert json.loads(resp["body"])["error"] == "internal_error"

    def test_missing_input_bucket_maps_to_500(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("INPUT_BUCKET_NAME")
        resp = handler.lambda_handler(make_event(VALID_BODY), None)
        assert resp["statusCode"] == 500
