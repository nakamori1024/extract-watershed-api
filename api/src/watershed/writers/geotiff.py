"""GeoTIFF output.

Write binary mask of watershed extraction (clipped by bbox) to GeoTIFF using rasterio.
Set CRS, transform, and deflate compression. In Lambda, specify /tmp path for output_path.
"""

import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

from watershed.models import BasinResult


def write_geotiff(result: BasinResult, output_path: str) -> None:
    """Write watershed mask to GeoTIFF file.

    Since nodata=0 is set, cells outside the basin are displayed transparently in QGIS, etc.

    Args:
        result: Watershed extraction result (mask / crs / transform)
        output_path: Output GeoTIFF path (e.g., /tmp/basin.tif)
    """
    mask = result.mask
    height, width = mask.shape

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint8",
        crs=CRS.from_user_input(result.crs),
        transform=Affine(*result.transform),
        nodata=0,
        compress="deflate",
    ) as dst:
        dst.write(mask, 1)
