"""Tests for writers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.crs import CRS
from rasterio.transform import Affine
from watershed.models import BasinResult, BoundingBox, PourPoint
from watershed.writers import write_geotiff, write_png


@pytest.fixture
def basin_result() -> BasinResult:
    """Result with a small 3×4 watershed mask (L-shaped)."""
    mask = np.array(
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
        ],
        dtype=np.uint8,
    )
    return BasinResult(
        mask=mask,
        bbox=BoundingBox(
            min_row=100,
            max_row=102,
            min_col=200,
            max_col=203,
            south=35.897,
            north=35.9,
            west=139.2,
            east=139.204,
        ),
        cell_count=6,
        area_km2=0.06,
        crs="EPSG:4326",
        transform=(0.001, 0.0, 139.2, 0.0, -0.001, 35.9),
        pour_point=PourPoint(35.8985, 139.2005),
        bfs_mode="chunked",
    )


class TestWriteGeotiff:
    def test_roundtrip(self, basin_result: BasinResult, tmp_path: Path) -> None:
        """Mask, CRS, and transform can be restored from written GeoTIFF."""
        out = str(tmp_path / "basin.tif")
        write_geotiff(basin_result, out)

        with rasterio.open(out) as src:
            assert src.crs == CRS.from_epsg(4326)
            assert src.transform == Affine(*basin_result.transform)
            assert src.nodata == 0
            assert src.profile["compress"] == "deflate"
            np.testing.assert_array_equal(src.read(1), basin_result.mask)


class TestWritePng:
    def test_rgba_values(self, basin_result: BasinResult, tmp_path: Path) -> None:
        """Basin cells are semi-transparent blue, outside basin is fully transparent."""
        out = str(tmp_path / "basin.png")
        write_png(basin_result, out)

        img = Image.open(out)
        assert img.mode == "RGBA"
        assert img.size == (4, 3)  # PIL is (width, height)
        rgba = np.asarray(img)
        # Basin cell (0,0)
        assert tuple(rgba[0, 0]) == (0, 100, 255, 128)
        # Outside basin cell (0,3): fully transparent
        assert tuple(rgba[0, 3]) == (0, 0, 0, 0)
        # Mask and transparency state match
        np.testing.assert_array_equal((rgba[:, :, 3] > 0), basin_result.mask == 1)
