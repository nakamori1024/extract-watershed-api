"""Mosaic J-FlwDir tiles into a single GeoTIFF.

J-FlwDir is distributed as 1 deg x 1 deg GeoTIFF tiles (3600 x 3600 px).
Since tiles share the same CRS, resolution, and are grid-aligned,
no resampling is needed -- each tile is simply written to its
corresponding window in the output.

rasterio.merge.merge is avoided because it materializes the entire
output in memory (~7.7 GB for all of Japan at uint8).  Instead, tiles
are streamed one at a time.  Non-aligned tiles cause an error.

Areas not covered by any tile are filled with *fill* (default 247 =
ocean in J-FlwDir; 0 means "river mouth" and must NOT be used as fill).
The output is explicitly initialized with the fill value because
unwritten (sparse) GTiff blocks would read back as 0.

The fill value and the NoData tag are separate concerns: the original
J-FlwDir tiles carry NoData=-9, which no uint8 pixel can match, so
effectively no pixel is nodata.  Following that behavior, no NoData tag
is written by default (*nodata=None*).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import rasterio.windows
from rasterio.transform import from_origin

# Grid-alignment tolerance in pixel units.
ALIGNMENT_TOLERANCE = 1e-6


def mosaic_tiles(
    input_dir: str,
    output_path: str,
    *,
    pattern: str = "*.tif",
    fill: int = 247,
    nodata: int | None = None,
    overwrite: bool = False,
) -> tuple[int, int, int]:
    """Mosaic GeoTIFF tiles in a directory into a single GeoTIFF.

    Args:
        input_dir: Directory containing the tiles.
        output_path: Path for the output GeoTIFF.
        pattern: Glob pattern for tile filenames (e.g. "*_dir.tif").
        fill: Fill value for areas with no tile coverage
            (247 = ocean in J-FlwDir).
        nodata: NoData tag for the output.  None (default) writes no tag,
            matching the effective behavior of the original tiles.
        overwrite: Overwrite the output if it already exists.

    Returns:
        (height, width, number_of_tiles)

    Raises:
        FileNotFoundError: No tiles match the pattern.
        FileExistsError: Output exists and *overwrite* is False.
        ValueError: Tiles have inconsistent CRS, resolution, or dtype,
            or are not grid-aligned.
    """
    tile_paths = sorted(Path(input_dir).glob(pattern))
    if not tile_paths:
        raise FileNotFoundError(f"No tiles found: {input_dir} (pattern={pattern})")

    if Path(output_path).exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    # Read headers only to validate consistency and compute extent.
    with rasterio.open(tile_paths[0]) as first:
        crs = first.crs
        xres, yres = first.res
        dtype = first.dtypes[0]

    left, bottom, right, top = None, None, None, None
    for path in tile_paths:
        with rasterio.open(path) as src:
            tile_xres, tile_yres = src.res
            if src.crs != crs:
                raise ValueError(f"CRS mismatch: {path} ({src.crs} != {crs})")
            if abs(tile_xres - xres) > ALIGNMENT_TOLERANCE * xres or (
                abs(tile_yres - yres) > ALIGNMENT_TOLERANCE * yres
            ):
                raise ValueError(f"Resolution mismatch: {path}")
            # b, d are rotation terms of the affine transform (normally 0).
            if src.transform.b != 0 or src.transform.d != 0:
                raise ValueError(f"Rotated transforms are not supported: {path}")
            if src.dtypes[0] != dtype:
                raise ValueError(f"dtype mismatch: {path}")
            bounds = src.bounds
            left = bounds.left if left is None else min(left, bounds.left)
            bottom = bounds.bottom if bottom is None else min(bottom, bounds.bottom)
            right = bounds.right if right is None else max(right, bounds.right)
            top = bounds.top if top is None else max(top, bounds.top)

    assert left is not None and bottom is not None
    assert right is not None and top is not None
    width = round((right - left) / xres)
    height = round((top - bottom) / yres)

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": from_origin(left, top, xres, yres),
        "nodata": nodata,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "compress": "deflate",
        "bigtiff": "IF_SAFER",
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        # Initialize the full extent with the fill value first: unwritten
        # (sparse) GTiff blocks read back as 0, which in J-FlwDir means
        # "river mouth", not ocean.  Uniform strips compress to almost
        # nothing under deflate, so the cost is negligible.
        strip_rows = 512
        fill_strip = np.full((strip_rows, width), fill, dtype=dtype)
        for row_start in range(0, height, strip_rows):
            row_end = min(row_start + strip_rows, height)
            strip_window = rasterio.windows.Window(
                0,  # type: ignore[call-arg]
                row_start,
                width,
                row_end - row_start,
            )
            dst.write(fill_strip[: row_end - row_start], 1, window=strip_window)

        # Write each tile into its corresponding window.
        for path in tile_paths:
            with rasterio.open(path) as src:
                col_f = (src.bounds.left - left) / xres
                row_f = (top - src.bounds.top) / yres
                col, row = round(col_f), round(row_f)
                if abs(col_f - col) > ALIGNMENT_TOLERANCE or abs(row_f - row) > (
                    ALIGNMENT_TOLERANCE
                ):
                    raise ValueError(f"Tile is not grid-aligned: {path}")
                window = rasterio.windows.Window(col, row, src.width, src.height)  # type: ignore[call-arg]
                dst.write(src.read(1), 1, window=window)

    return height, width, len(tile_paths)
