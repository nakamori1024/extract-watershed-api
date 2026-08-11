"""Tests for convert_to_cog."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import rasterio
from converter.to_cog import convert_to_cog


def read_band(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        result: np.ndarray = src.read(1)
    return result


class TestBasicConversion:
    def test_data_is_identical(self, source_tif: str, tmp_path: Path) -> None:
        out = str(tmp_path / "out_cog.tif")
        convert_to_cog(source_tif, out)
        np.testing.assert_array_equal(read_band(out), read_band(source_tif))

    def test_default_block_size_512(self, source_tif: str, tmp_path: Path) -> None:
        out = str(tmp_path / "out_cog.tif")
        convert_to_cog(source_tif, out)
        with rasterio.open(out) as src:
            assert src.block_shapes[0] == (512, 512)

    def test_custom_block_size(self, source_tif: str, tmp_path: Path) -> None:
        out = str(tmp_path / "out_cog.tif")
        convert_to_cog(source_tif, out, block_size=32)
        with rasterio.open(out) as src:
            assert src.block_shapes[0] == (32, 32)
        np.testing.assert_array_equal(read_band(out), read_band(source_tif))

    def test_default_compress_lzw(self, source_tif: str, tmp_path: Path) -> None:
        """Default compression is LZW."""
        out = str(tmp_path / "out_cog.tif")
        convert_to_cog(source_tif, out)
        with rasterio.open(out) as src:
            assert src.profile["compress"] == "lzw"

    def test_no_overviews(self, source_tif: str, tmp_path: Path) -> None:
        out = str(tmp_path / "out_cog.tif")
        convert_to_cog(source_tif, out, overviews=False)
        with rasterio.open(out) as src:
            assert src.overviews(1) == []

    def test_georeferencing_preserved(self, source_tif: str, tmp_path: Path) -> None:
        out = str(tmp_path / "out_cog.tif")
        convert_to_cog(source_tif, out)
        with rasterio.open(source_tif) as src, rasterio.open(out) as dst:
            assert dst.crs == src.crs
            assert dst.transform == src.transform
            assert dst.nodata == src.nodata


@pytest.mark.skipif(shutil.which("gdalinfo") is None, reason="gdalinfo not available")
class TestGdalInterop:
    def test_gdalinfo_reports_cog_layout(self, source_tif: str, tmp_path: Path) -> None:
        """Verify the output has COG layout via GDAL's own gdalinfo."""
        out = str(tmp_path / "out_cog.tif")
        convert_to_cog(source_tif, out)
        info = json.loads(
            subprocess.run(
                ["gdalinfo", "-json", out], capture_output=True, check=True, text=True
            ).stdout
        )
        assert info["metadata"]["IMAGE_STRUCTURE"]["LAYOUT"] == "COG"


class TestErrorHandling:
    def test_missing_input_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            convert_to_cog(str(tmp_path / "nope.tif"), str(tmp_path / "out.tif"))

    def test_block_size_not_multiple_of_16_raises(
        self, source_tif: str, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="multiple of 16"):
            convert_to_cog(source_tif, str(tmp_path / "out.tif"), block_size=100)

    def test_existing_output_without_overwrite_raises(
        self, source_tif: str, tmp_path: Path
    ) -> None:
        out = str(tmp_path / "out_cog.tif")
        convert_to_cog(source_tif, out)
        with pytest.raises(FileExistsError):
            convert_to_cog(source_tif, out)

    def test_overwrite_flag_allows_reconversion(
        self, source_tif: str, tmp_path: Path
    ) -> None:
        out = str(tmp_path / "out_cog.tif")
        convert_to_cog(source_tif, out, block_size=32)
        convert_to_cog(source_tif, out, block_size=64, overwrite=True)
        with rasterio.open(out) as src:
            assert src.block_shapes[0] == (64, 64)
