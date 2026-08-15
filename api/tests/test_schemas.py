"""Tests for schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from watershed.service.schemas import ExtractRequest


class TestExtractRequest:
    def test_valid_request(self) -> None:
        req = ExtractRequest(latitude=35.7446, longitude=140.8507, format="zarr")
        assert req.dataset is None

    def test_dataset_variant_key(self) -> None:
        req = ExtractRequest(
            latitude=35.0,
            longitude=140.0,
            format="cog",
            dataset="jflwdir-bench/dir_cog_b512.tif",
        )
        assert req.dataset == "jflwdir-bench/dir_cog_b512.tif"

    @pytest.mark.parametrize(
        ("latitude", "longitude"),
        [
            (23.9, 140.0),  # Out of bounds to the south
            (46.1, 140.0),  # Out of bounds to the north
            (35.0, 121.9),  # Out of bounds to the west
            (35.0, 149.1),  # Out of bounds to the east
        ],
    )
    def test_out_of_japan_bounds_rejected(
        self, latitude: float, longitude: float
    ) -> None:
        with pytest.raises(ValidationError):
            ExtractRequest(latitude=latitude, longitude=longitude, format="cog")

    def test_unknown_format_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ExtractRequest(latitude=35.0, longitude=140.0, format="netcdf")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "dataset",
        [
            "/absolute/path.tif",  # Absolute path
            "../other-prefix/data.tif",  # Path traversal
            "key with space.tif",  # Invalid characters
        ],
    )
    def test_invalid_dataset_rejected(self, dataset: str) -> None:
        with pytest.raises(ValidationError):
            ExtractRequest(
                latitude=35.0, longitude=140.0, format="cog", dataset=dataset
            )
