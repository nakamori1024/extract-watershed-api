"""Writers for watershed extraction results (GeoTIFF / PNG)."""

from watershed.writers.geotiff import write_geotiff
from watershed.writers.png import write_png

__all__ = ["write_geotiff", "write_png"]
