"""GeoTIFF to COG conversion.

Conversion is delegated to the GDAL COG driver (via rasterio's bundled GDAL),
equivalent to ``gdal_translate -of COG``.  Block size is an experimental
variable for benchmarking, so it is exposed as a parameter.
"""

from __future__ import annotations

from pathlib import Path

import rasterio
import rasterio.shutil


def convert_to_cog(
    input_path: str,
    output_path: str,
    *,
    block_size: int = 512,
    compress: str = "LZW",
    overviews: bool = True,
    overwrite: bool = False,
) -> None:
    """Convert a GeoTIFF to COG format.

    Args:
        input_path: Path to the input GeoTIFF file.
        output_path: Path for the output COG file.
        block_size: Internal tile size in pixels (must be a multiple of 16
            per TIFF spec).
        compress: Compression method (LZW / DEFLATE / ZSTD / NONE).
        overviews: Whether to generate overviews.  Downsampling uses NEAREST
            because D8 flow direction codes are categorical values and
            interpolation would produce invalid values.
        overwrite: Overwrite the output if it already exists.

    Raises:
        FileNotFoundError: Input file does not exist.
        FileExistsError: Output exists and *overwrite* is False.
        ValueError: *block_size* is not a multiple of 16.
    """
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if block_size % 16 != 0:
        raise ValueError(
            f"block_size must be a multiple of 16: block_size={block_size}"
        )

    if Path(output_path).exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    with rasterio.open(input_path) as src:
        rasterio.shutil.copy(
            src,
            output_path,
            driver="COG",
            blocksize=block_size,
            compress=compress,
            overviews="AUTO" if overviews else "NONE",
            overview_resampling="NEAREST",
            num_threads="ALL_CPUS",
            bigtiff="IF_SAFER",
        )
