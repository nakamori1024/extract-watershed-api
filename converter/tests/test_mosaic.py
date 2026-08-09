"""Tests for mosaic_tiles.

Uses small tiles (1 deg = 10 px, resolution 0.1 deg) that mimic
the J-FlwDir distribution format (1 deg x 1 deg GeoTIFF tiles).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from converter.mosaic import mosaic_tiles
from rasterio.transform import from_origin

RES = 0.1
# Tile side length in pixels, derived from resolution (RES=0.1 -> 10 px).
SIZE = round(1.0 / RES)


def make_tile(
    dir_path: Path,
    lon: int,
    lat: int,
    value: int,
    *,
    crs: str = "EPSG:4326",
    res: float = RES,
    shift: float = 0.0,
) -> Path:
    """Create a 1 deg x 1 deg tile at lower-left (lon, lat) using J-FlwDir naming."""
    path = dir_path / f"n{lat:02d}e{lon:03d}_dir.tif"
    size = round(1.0 / res)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=1,
        dtype="uint8",
        crs=crs,
        transform=from_origin(lon + shift, lat + 1, res, res),
    ) as dst:
        dst.write(np.full((size, size), value, dtype=np.uint8), 1)
    return path


class TestMosaic:
    def test_2x2_tiles(self, tmp_path: Path) -> None:
        tiles = tmp_path / "tiles"
        tiles.mkdir()
        # NW=1, NE=2, SW=3, SE=4
        make_tile(tiles, 139, 35, 1)
        make_tile(tiles, 140, 35, 2)
        make_tile(tiles, 139, 34, 3)
        make_tile(tiles, 140, 34, 4)
        out = str(tmp_path / "mosaic.tif")

        height, width, n_tiles = mosaic_tiles(str(tiles), out)

        assert (height, width, n_tiles) == (2 * SIZE, 2 * SIZE, 4)
        with rasterio.open(out) as src:
            assert src.transform.c == 139.0  # west edge
            assert src.transform.f == 36.0  # north edge
            data = src.read(1)
        assert (data[:SIZE, :SIZE] == 1).all()
        assert (data[:SIZE, SIZE:] == 2).all()
        assert (data[SIZE:, :SIZE] == 3).all()
        assert (data[SIZE:, SIZE:] == 4).all()

    def test_missing_tile_filled_with_ocean_value(self, tmp_path: Path) -> None:
        tiles = tmp_path / "tiles"
        tiles.mkdir()
        make_tile(tiles, 139, 35, 1)
        make_tile(tiles, 140, 35, 2)
        make_tile(tiles, 139, 34, 3)  # no SE tile
        out = str(tmp_path / "mosaic.tif")

        mosaic_tiles(str(tiles), out)

        with rasterio.open(out) as src:
            assert src.nodata is None
            data = src.read(1)
        assert (data[SIZE:, SIZE:] == 247).all()
        assert (data[:SIZE, :SIZE] == 1).all()

    def test_explicit_nodata_tag(self, tmp_path: Path) -> None:
        tiles = tmp_path / "tiles"
        tiles.mkdir()
        make_tile(tiles, 139, 35, 1)
        out = str(tmp_path / "mosaic.tif")

        mosaic_tiles(str(tiles), out, nodata=247)

        with rasterio.open(out) as src:
            assert src.nodata == 247

    def test_pattern_filters_files(self, tmp_path: Path) -> None:
        tiles = tmp_path / "tiles"
        tiles.mkdir()
        make_tile(tiles, 139, 35, 1)
        # J-FlwDir ships other layers (e.g. elv) alongside dir tiles.
        (tiles / "n35e139_elv.tif").write_bytes(b"dummy")
        out = str(tmp_path / "mosaic.tif")

        _, _, n_tiles = mosaic_tiles(str(tiles), out, pattern="*_dir.tif")

        assert n_tiles == 1


class TestValidation:
    def test_no_tiles_raises(self, tmp_path: Path) -> None:
        tiles = tmp_path / "empty"
        tiles.mkdir()
        with pytest.raises(FileNotFoundError):
            mosaic_tiles(str(tiles), str(tmp_path / "out.tif"))

    def test_crs_mismatch_raises(self, tmp_path: Path) -> None:
        tiles = tmp_path / "tiles"
        tiles.mkdir()
        make_tile(tiles, 139, 35, 1)
        make_tile(tiles, 140, 35, 2, crs="EPSG:4612")
        with pytest.raises(ValueError, match="CRS"):
            mosaic_tiles(str(tiles), str(tmp_path / "out.tif"))

    def test_resolution_mismatch_raises(self, tmp_path: Path) -> None:
        tiles = tmp_path / "tiles"
        tiles.mkdir()
        make_tile(tiles, 139, 35, 1)
        make_tile(tiles, 140, 35, 2, res=0.2)
        with pytest.raises(ValueError, match="Resolution mismatch"):
            mosaic_tiles(str(tiles), str(tmp_path / "out.tif"))

    def test_unaligned_tile_raises(self, tmp_path: Path) -> None:
        tiles = tmp_path / "tiles"
        tiles.mkdir()
        make_tile(tiles, 139, 35, 1)
        make_tile(tiles, 140, 35, 2, shift=RES / 2)  # half-pixel offset
        with pytest.raises(ValueError, match="not grid-aligned"):
            mosaic_tiles(str(tiles), str(tmp_path / "out.tif"))

    def test_existing_output_without_overwrite_raises(self, tmp_path: Path) -> None:
        tiles = tmp_path / "tiles"
        tiles.mkdir()
        make_tile(tiles, 139, 35, 1)
        out = str(tmp_path / "mosaic.tif")
        mosaic_tiles(str(tiles), out)
        with pytest.raises(FileExistsError):
            mosaic_tiles(str(tiles), out)
