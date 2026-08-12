"""Shared fixtures for api tests.

Prepares a flow direction GeoTIFF that can be converted to COG / Zarr
by converter, and verifies the contract that readers can consume what
converter produces.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from numpy.typing import NDArray
from rasterio.transform import from_origin

# Scaled-down version using the same CRS as J-FlwDir (EPSG:4326).
# Resolution is set to a round 0.001 degrees for easy hand-calculation.
ORIGIN_X = 139.0
ORIGIN_Y = 36.0
RES = 0.001


@pytest.fixture(scope="session")
def flow_grid() -> NDArray[np.uint8]:
    """Random D8 flow direction field (100 rows x 150 cols, fixed seed)."""
    rng = np.random.default_rng(42)
    grid = rng.choice(
        [1, 2, 4, 8, 16, 32, 64, 128, 0, 247, 255], size=(100, 150)
    ).astype(np.uint8)
    # Guarantee a cell with a valid flow code for use as a pour point
    grid[50, 75] = 1
    return grid


@pytest.fixture(scope="session")
def flow_tif(
    flow_grid: NDArray[np.uint8], tmp_path_factory: pytest.TempPathFactory
) -> str:
    """GeoTIFF containing the flow direction field (converter input format)."""
    path = tmp_path_factory.mktemp("data") / "flow_dir.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=flow_grid.shape[0],
        width=flow_grid.shape[1],
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(ORIGIN_X, ORIGIN_Y, RES, RES),
    ) as dst:
        dst.write(flow_grid, 1)
    return str(path)


@pytest.fixture(scope="session")
def zarr_store(flow_tif: str, tmp_path_factory: pytest.TempPathFactory) -> str:
    """Zarr store converted by converter (chunk size 32)."""
    from converter.to_zarr import convert_geotiff_to_zarr

    out = str(tmp_path_factory.mktemp("data") / "flow.zarr")
    convert_geotiff_to_zarr(flow_tif, out, chunk_size=32, fill_value=247)
    return out


@pytest.fixture(scope="session")
def cog_file(flow_tif: str, tmp_path_factory: pytest.TempPathFactory) -> str:
    """COG converted by converter (block size 32)."""
    from converter.to_cog import convert_to_cog

    out = str(tmp_path_factory.mktemp("data") / "flow_cog.tif")
    convert_to_cog(flow_tif, out, block_size=32, overviews=False)
    return out


@pytest.fixture(scope="session")
def sharded_zarr_store(flow_tif: str, tmp_path_factory: pytest.TempPathFactory) -> str:
    """Sharded Zarr store (chunk size 16 / shard size 32)."""
    from converter.to_zarr import convert_geotiff_to_zarr

    out = str(tmp_path_factory.mktemp("data") / "flow_sharded.zarr")
    convert_geotiff_to_zarr(flow_tif, out, chunk_size=16, shard_size=32, fill_value=247)
    return out
