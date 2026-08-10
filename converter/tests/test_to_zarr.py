"""Tests for convert_geotiff_to_zarr."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import rasterio
import rasterio.crs
import zarr
import zarr.errors
from converter.to_zarr import convert_geotiff_to_zarr
from rasterio.transform import from_origin


def read_source(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        result: np.ndarray = src.read(1)
    return result


def open_main_array(store_path: str) -> zarr.Array:
    """Open the main array (named after the output path stem) from a Zarr store."""
    group = zarr.open_group(store_path, mode="r")
    array = group[Path(store_path).stem]
    assert isinstance(array, zarr.Array)
    return array


class TestBasicConversion:
    def test_data_is_identical(self, source_tif: str, tmp_path: Path) -> None:
        out = str(tmp_path / "out.zarr")
        convert_geotiff_to_zarr(source_tif, out)
        z = open_main_array(out)
        np.testing.assert_array_equal(z[:], read_source(source_tif))

    def test_dtype_is_uint8(self, source_tif: str, tmp_path: Path) -> None:
        out = str(tmp_path / "out.zarr")
        convert_geotiff_to_zarr(source_tif, out)
        assert open_main_array(out).dtype == np.uint8

    def test_default_chunk_size(self, source_tif: str, tmp_path: Path) -> None:
        out = str(tmp_path / "out.zarr")
        convert_geotiff_to_zarr(source_tif, out)
        assert open_main_array(out).chunks == (512, 512)


class TestGdalConventionMetadata:
    """Verify metadata follows GDAL's Zarr driver conventions."""

    def test_crs_as_wkt(self, source_tif: str, tmp_path: Path) -> None:
        """Input GeoTIFF CRS is stored as WKT."""
        out = str(tmp_path / "out.zarr")
        convert_geotiff_to_zarr(source_tif, out)
        crs_attr = open_main_array(out).attrs["_CRS"]
        assert isinstance(crs_attr, dict)
        with rasterio.open(source_tif) as src:
            assert rasterio.crs.CRS.from_wkt(crs_attr["wkt"]) == src.crs

    def test_nodata_as_fill_value(self, source_tif: str, tmp_path: Path) -> None:
        """fill_value defaults to the input's nodata tag."""
        out = str(tmp_path / "out.zarr")
        convert_geotiff_to_zarr(source_tif, out)
        assert open_main_array(out).fill_value == 0

    def test_explicit_fill_value(self, source_tif: str, tmp_path: Path) -> None:
        """In production, J-FlwDir uses 247 (ocean) as fill_value."""
        out = str(tmp_path / "out.zarr")
        convert_geotiff_to_zarr(source_tif, out, fill_value=247)
        assert open_main_array(out).fill_value == 247

    def test_coordinate_arrays_are_pixel_centers(
        self, source_tif: str, tmp_path: Path
    ) -> None:
        out = str(tmp_path / "out.zarr")
        convert_geotiff_to_zarr(source_tif, out)
        group = zarr.open_group(out, mode="r")
        x, y = group["X"], group["Y"]
        assert isinstance(x, zarr.Array) and isinstance(y, zarr.Array)
        assert x.shape == (150,) and y.shape == (100,)
        # Pixel-center values: origin + resolution * (index + 0.5).
        # Expected values hand-calculated from conftest source_tif (origin 139/36, res 0.001)
        np.testing.assert_allclose(np.asarray(x[0:2]), [139.0005, 139.0015])
        np.testing.assert_allclose(np.asarray(y[0:2]), [35.9995, 35.9985])

    def test_dimension_names(self, source_tif: str, tmp_path: Path) -> None:
        out = str(tmp_path / "out.zarr")
        convert_geotiff_to_zarr(source_tif, out)
        meta = json.loads((Path(out) / "out" / "zarr.json").read_text())
        assert meta["dimension_names"] == ["Y", "X"]

    def test_nodata_none_becomes_zero(self, tmp_path: Path) -> None:
        """Input without a nodata tag defaults fill_value to 0."""
        path = tmp_path / "no_nodata.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=10,
            width=10,
            count=1,
            dtype="uint8",
            crs="EPSG:4326",
            transform=from_origin(139.0, 36.0, 0.001, 0.001),
        ) as dst:
            dst.write(np.ones((10, 10), dtype=np.uint8), 1)
        out = str(tmp_path / "out.zarr")
        convert_geotiff_to_zarr(str(path), out)
        assert open_main_array(out).fill_value == 0


@pytest.mark.skipif(shutil.which("gdalinfo") is None, reason="gdalinfo not available")
class TestGdalInterop:
    """Verify GDAL can open the store with full georeferencing."""

    def test_gdalinfo_reads_georeferencing(
        self, source_tif: str, tmp_path: Path
    ) -> None:
        out = str(tmp_path / "out.zarr")
        convert_geotiff_to_zarr(source_tif, out)
        info = json.loads(
            subprocess.run(
                ["gdalinfo", "-json", out], capture_output=True, check=True, text=True
            ).stdout
        )
        assert info["size"] == [150, 100]
        np.testing.assert_allclose(
            info["geoTransform"], [139.0, 0.001, 0.0, 36.0, 0.0, -0.001]
        )
        assert "4326" in json.dumps(info["stac"]) or "WGS 84" in json.dumps(
            info["coordinateSystem"]
        )
        assert info["bands"][0]["noDataValue"] == 0


class TestChunkAndShardOptions:
    def test_custom_chunk_size(self, source_tif: str, tmp_path: Path) -> None:
        out = str(tmp_path / "out.zarr")
        convert_geotiff_to_zarr(source_tif, out, chunk_size=64)
        z = open_main_array(out)
        assert z.chunks == (64, 64)
        np.testing.assert_array_equal(z[:], read_source(source_tif))

    def test_sharding(self, source_tif: str, tmp_path: Path) -> None:
        out = str(tmp_path / "out.zarr")
        convert_geotiff_to_zarr(source_tif, out, chunk_size=32, shard_size=64)
        z = open_main_array(out)
        assert z.chunks == (32, 32)
        assert z.shards == (64, 64)
        np.testing.assert_array_equal(z[:], read_source(source_tif))

    def test_shard_not_multiple_of_chunk_raises(
        self, source_tif: str, tmp_path: Path
    ) -> None:
        out = str(tmp_path / "out.zarr")
        with pytest.raises(ValueError, match="multiple of chunk_size"):
            convert_geotiff_to_zarr(source_tif, out, chunk_size=32, shard_size=50)


class TestErrorHandling:
    def test_missing_input_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            convert_geotiff_to_zarr(
                str(tmp_path / "nope.tif"), str(tmp_path / "out.zarr")
            )

    def test_non_geotiff_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "not_a_tif.tif"
        bad.write_bytes(b"this is not a geotiff")
        with pytest.raises(ValueError, match="GeoTIFF"):
            convert_geotiff_to_zarr(str(bad), str(tmp_path / "out.zarr"))

    def test_existing_output_without_overwrite_raises(
        self, source_tif: str, tmp_path: Path
    ) -> None:
        out = str(tmp_path / "out.zarr")
        convert_geotiff_to_zarr(source_tif, out)
        with pytest.raises(zarr.errors.ContainsGroupError):
            convert_geotiff_to_zarr(source_tif, out)

    def test_overwrite_flag_allows_reconversion(
        self, source_tif: str, tmp_path: Path
    ) -> None:
        out = str(tmp_path / "out.zarr")
        convert_geotiff_to_zarr(source_tif, out, chunk_size=64)
        convert_geotiff_to_zarr(source_tif, out, chunk_size=32, overwrite=True)
        assert open_main_array(out).chunks == (32, 32)
