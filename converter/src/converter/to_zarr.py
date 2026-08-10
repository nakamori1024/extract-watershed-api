"""GeoTIFF to Zarr v3 conversion.

Reads a GeoTIFF (typically the mosaic output) via rasterio and writes it
as a Zarr v3 store.  Metadata follows GDAL's own Zarr driver conventions
(no custom keys):

- CRS          → stored as WKT in the ``_CRS`` attribute of the main array
- Geolocation  → 1-D coordinate arrays ``X`` / ``Y`` (pixel-center values)
                  + ``dimension_names``

The resulting store can be opened directly in gdalinfo / QGIS with full
georeferencing.  However, the sharded variant is not yet supported by
GDAL (as of 3.11: ``ERROR 6: Unsupported codec: sharding_indexed``).

Chunk size, shard size, and zstd compression level are experimental
variables for benchmarking, so all are exposed as parameters.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import rasterio.errors
import rasterio.windows
import zarr
from zarr.codecs import ZstdCodec


def convert_geotiff_to_zarr(
    input_path: str,
    output_path: str,
    *,
    chunk_size: int = 512,
    shard_size: int | None = None,
    zstd_level: int = 0,
    fill_value: int | None = None,
    overwrite: bool = False,
) -> zarr.Array:
    """Convert a GeoTIFF file to Zarr v3 format.

    The output is a group-structured store containing the main array
    (named after the output path stem) and coordinate arrays ``X`` / ``Y``.

    Args:
        input_path: Path to the input GeoTIFF file.
        output_path: Path for the output Zarr store.
        chunk_size: Chunk side length in pixels.
        shard_size: Shard side length in pixels (must be a multiple of
            *chunk_size*).  None disables sharding.
        zstd_level: zstd compression level (0 is the zarr default).
        fill_value: Value for unwritten chunks.  None falls back to the
            input's nodata tag, then to 0 (247 = ocean is recommended
            for J-FlwDir).
        overwrite: Overwrite the output if it already exists.

    Returns:
        The written main array (zarr.Array).

    Raises:
        FileNotFoundError: Input file does not exist.
        ValueError: Input is not a GeoTIFF, has a rotated transform,
            or *shard_size* is not a multiple of *chunk_size*.
    """
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if shard_size is not None and shard_size % chunk_size != 0:
        raise ValueError(
            f"shard_size must be a multiple of chunk_size: "
            f"shard_size={shard_size}, chunk_size={chunk_size}"
        )

    try:
        src = rasterio.open(input_path)
    except rasterio.errors.RasterioIOError as e:
        raise ValueError(f"Input file is not a GeoTIFF: {input_path}") from e

    with src:
        if src.driver != "GTiff":
            raise ValueError(f"Input file is not a GeoTIFF: {input_path}")

        # b, d are the rotation terms of the affine transform (always 0 for standard rasters)
        if src.transform.b != 0 or src.transform.d != 0:
            raise ValueError(f"Rotated transforms are not supported: {input_path}")

        height, width = src.height, src.width
        if fill_value is None:
            fill_value = int(src.nodata) if src.nodata is not None else 0

        root = zarr.create_group(store=output_path, overwrite=overwrite)

        z = root.create_array(
            name=Path(output_path).stem,
            shape=(height, width),
            chunks=(chunk_size, chunk_size),
            shards=(shard_size, shard_size) if shard_size is not None else None,
            dtype="uint8",
            compressors=ZstdCodec(level=zstd_level),
            fill_value=fill_value,
            dimension_names=["Y", "X"],
        )
        z.attrs["_CRS"] = {"wkt": src.crs.to_wkt()}

        # Coordinate arrays (pixel-center values), matching GDAL's Zarr driver convention
        xres, yres = src.res
        for name, values in (
            ("X", src.bounds.left + xres * (np.arange(width) + 0.5)),
            ("Y", src.bounds.top - yres * (np.arange(height) + 0.5)),
        ):
            coord = root.create_array(
                name=name,
                shape=values.shape,
                chunks=values.shape,
                dtype="float64",
                compressors=None,
                fill_value=np.nan,
                dimension_names=[name],
            )
            coord[:] = values

        # When sharding is enabled, write in shard-sized blocks to avoid
        # repeated partial writes to the same shard.
        block = shard_size if shard_size is not None else chunk_size

        # Scan row-wise using windowed reads
        for row_start in range(0, height, block):
            row_end = min(row_start + block, height)
            for col_start in range(0, width, block):
                col_end = min(col_start + block, width)
                window = rasterio.windows.Window(
                    col_start,  # type: ignore[call-arg]
                    row_start,
                    col_end - col_start,
                    row_end - row_start,
                )
                data = src.read(1, window=window).astype(np.uint8)
                z[row_start:row_end, col_start:col_end] = data

    return z
