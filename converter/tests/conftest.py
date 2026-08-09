"""Shared test fixtures for the converter package."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

# Same CRS as J-FlwDir (EPSG:4326), with reduced dimensions for testing.
CRS = "EPSG:4326"

# Resolution is 0.001 deg instead of the actual 1 arc-second (1/3600 ≈ 0.000278 deg)
# to keep expected values easy to verify by hand.
TRANSFORM = from_origin(139.0, 36.0, 0.001, 0.001)


@pytest.fixture
def source_tif(tmp_path: Path) -> str:
    """Create a small GeoTIFF (100 x 150 px) for testing."""
    path = tmp_path / "flow_dir.tif"
    rng = np.random.default_rng(42)
    data = rng.choice(
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 247, 255], size=(100, 150)
    ).astype(np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=100,
        width=150,
        count=1,
        dtype="uint8",
        crs=CRS,
        transform=TRANSFORM,
        nodata=0,
    ) as dst:
        dst.write(data, 1)
    return str(path)
