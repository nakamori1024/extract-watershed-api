"""Tests for readers.

Verifies that ZarrReader / CogReader correctly read converter output,
and that both readers behave identically as FlowReader implementations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from numpy.typing import NDArray
from rasterio.transform import from_origin
from watershed.extractor import extract_basin
from watershed.models import PourPoint
from watershed.readers import CogReader, ZarrReader

# Same values as the flow_tif fixture in conftest.py
ORIGIN_X = 139.0
ORIGIN_Y = 36.0
RES = 0.001

EXPECTED_TRANSFORM = (RES, 0.0, ORIGIN_X, 0.0, -RES, ORIGIN_Y)


def make_reader(kind: str, zarr_store: str, cog_file: str) -> ZarrReader | CogReader:
    return ZarrReader(zarr_store) if kind == "zarr" else CogReader(cog_file)


@pytest.mark.parametrize("kind", ["zarr", "cog"])
class TestFlowReaderContract:
    """FlowReader contract shared by both readers."""

    def test_crs(self, kind: str, zarr_store: str, cog_file: str) -> None:
        reader = make_reader(kind, zarr_store, cog_file)
        assert reader.crs == "EPSG:4326"

    def test_transform(self, kind: str, zarr_store: str, cog_file: str) -> None:
        reader = make_reader(kind, zarr_store, cog_file)
        np.testing.assert_allclose(reader.transform, EXPECTED_TRANSFORM)

    def test_shape_and_chunk_shape(
        self, kind: str, zarr_store: str, cog_file: str
    ) -> None:
        reader = make_reader(kind, zarr_store, cog_file)
        assert reader.shape == (100, 150)
        assert reader.chunk_shape == (32, 32)

    def test_get_cell_matches_source(
        self,
        kind: str,
        zarr_store: str,
        cog_file: str,
        flow_grid: NDArray[np.uint8],
    ) -> None:
        reader = make_reader(kind, zarr_store, cog_file)
        for row, col in [(0, 0), (50, 75), (99, 149), (31, 32)]:
            assert reader.get_cell(row, col) == int(flow_grid[row, col])

    def test_get_cell_out_of_bounds_raises(
        self, kind: str, zarr_store: str, cog_file: str
    ) -> None:
        reader = make_reader(kind, zarr_store, cog_file)
        with pytest.raises(IndexError):
            reader.get_cell(100, 0)
        with pytest.raises(IndexError):
            reader.get_cell(0, -1)

    def test_edge_chunk_is_truncated(
        self, kind: str, zarr_store: str, cog_file: str
    ) -> None:
        """Edge chunks are truncated to the actual data size (150 = 32*4 + 22)."""
        reader = make_reader(kind, zarr_store, cog_file)
        assert reader.get_chunk(0, 4).shape == (32, 22)
        assert reader.get_chunk(3, 0).shape == (4, 32)  # 100 = 32*3 + 4

    def test_latlon_to_pixel(self, kind: str, zarr_store: str, cog_file: str) -> None:
        reader = make_reader(kind, zarr_store, cog_file)
        # Center coordinate of pixel (50, 75)
        lat = ORIGIN_Y - (50 + 0.5) * RES
        lon = ORIGIN_X + (75 + 0.5) * RES
        assert reader.latlon_to_pixel(lat, lon) == (50, 75)

    def test_latlon_out_of_bounds_raises(
        self, kind: str, zarr_store: str, cog_file: str
    ) -> None:
        reader = make_reader(kind, zarr_store, cog_file)
        with pytest.raises(ValueError, match="outside the data extent"):
            reader.latlon_to_pixel(ORIGIN_Y + 1.0, ORIGIN_X + 0.05)

    def test_get_cell_uses_chunk_cache(
        self, kind: str, zarr_store: str, cog_file: str
    ) -> None:
        """get_cell reads at chunk granularity and caches (symmetric across both readers)."""
        reader = make_reader(kind, zarr_store, cog_file)
        reader.get_cell(50, 75)
        assert reader.cache_size == 1
        assert reader.get_chunk(50 // 32, 75 // 32) is not None

    def test_chunk_cache(self, kind: str, zarr_store: str, cog_file: str) -> None:
        """Re-fetching the same chunk returns from cache."""
        reader = make_reader(kind, zarr_store, cog_file)
        assert reader.cache_size == 0
        first = reader.get_chunk(1, 1)
        assert reader.cache_size == 1
        assert reader.get_chunk(1, 1) is first  # same object

    def test_prefetch_fills_cache(
        self, kind: str, zarr_store: str, cog_file: str
    ) -> None:
        reader = make_reader(kind, zarr_store, cog_file)
        reader.prefetch_chunks([(0, 0), (0, 1), (1, 0)])
        assert reader.cache_size == 3


class TestCrossFormatEquivalence:
    """Extraction results are identical between COG and Zarr (prerequisite for benchmark fairness)."""

    def test_extract_basin_identical(self, zarr_store: str, cog_file: str) -> None:
        lat = ORIGIN_Y - (50 + 0.5) * RES
        lon = ORIGIN_X + (75 + 0.5) * RES
        pour = PourPoint(lat, lon)

        result_zarr = extract_basin(ZarrReader(zarr_store), pour)
        with CogReader(cog_file) as cog_reader:
            result_cog = extract_basin(cog_reader, pour)

        assert result_zarr.cell_count == result_cog.cell_count
        # Pixel coordinates must match exactly; geographic coordinates are
        # compared with tolerance (Zarr's transform is reconstructed from
        # coordinate arrays, introducing ~1e-13 rounding differences)
        bz, bc = result_zarr.bbox, result_cog.bbox
        assert (bz.min_row, bz.max_row, bz.min_col, bz.max_col) == (
            bc.min_row,
            bc.max_row,
            bc.min_col,
            bc.max_col,
        )
        for field in ("north", "south", "west", "east"):
            assert getattr(bz, field) == pytest.approx(getattr(bc, field))
        assert result_zarr.area_km2 == pytest.approx(result_cog.area_km2)
        assert result_zarr.crs == result_cog.crs
        np.testing.assert_allclose(result_zarr.transform, result_cog.transform)
        np.testing.assert_array_equal(result_zarr.mask, result_cog.mask)


class TestShardedZarr:
    def test_chunk_shape_is_inner_chunk(self, sharded_zarr_store: str) -> None:
        """For sharded stores, the inner chunk is the read unit."""
        reader = ZarrReader(sharded_zarr_store)
        assert reader.chunk_shape == (16, 16)

    def test_data_matches_source(
        self, sharded_zarr_store: str, flow_grid: NDArray[np.uint8]
    ) -> None:
        reader = ZarrReader(sharded_zarr_store)
        for row, col in [(0, 0), (50, 75), (99, 149)]:
            assert reader.get_cell(row, col) == int(flow_grid[row, col])


class TestZarrReaderErrors:
    def test_missing_store_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            ZarrReader(str(tmp_path / "nope.zarr"))


class TestCogReaderErrors:
    def test_untiled_geotiff_raises(self, flow_tif: str) -> None:
        """Untiled (strip-layout) GeoTIFFs are rejected."""
        with pytest.raises(ValueError, match="tiled GeoTIFF"):
            CogReader(flow_tif)

    def test_non_epsg4326_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "utm.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=64,
            width=64,
            count=1,
            dtype="uint8",
            crs="EPSG:32654",
            transform=from_origin(400000, 4000000, 30, 30),
            tiled=True,
            blockxsize=32,
            blockysize=32,
        ) as dst:
            dst.write(np.ones((64, 64), dtype=np.uint8), 1)
        with pytest.raises(ValueError, match="EPSG:4326"):
            CogReader(str(path))
