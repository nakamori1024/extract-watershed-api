"""COG (Cloud Optimized GeoTIFF) flow direction data reader.

Provides block-level (internal tile) access via rasterio Window reads.
COGs on S3 are opened through /vsis3/ with GDAL environment variables
tuned for S3 access.

The design mirrors ZarrReader (chunk cache, prefetch, FlowReader
compatibility), except that prefetch is sequential rather than parallel
because rasterio's DatasetReader is not thread-safe.  This asymmetry is
a deliberate consideration for format comparison; changes here should
account for benchmark interpretation.
"""

import math
import os
from typing import Self

import numpy as np
import rasterio
from numpy.typing import NDArray
from rasterio.windows import Window

# GDAL environment variables to optimize COG access on S3
_GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "TRUE",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
}


class CogReader:
    """FlowReader implementation that reads flow direction data from a COG file.

    Args:
        cog_path: Local path or S3 URL (s3://bucket/key.tif).

    Raises:
        ValueError: If the CRS is not EPSG:4326, or the GeoTIFF is not
            internally tiled.
    """

    def __init__(self, cog_path: str) -> None:
        # Note: affects the entire process (harmless in Lambda)
        for key, value in _GDAL_ENV.items():
            os.environ[key] = value

        s3_path = cog_path
        if cog_path.startswith("s3://"):
            s3_path = cog_path.replace("s3://", "/vsis3/", 1)

        self._dataset: rasterio.DatasetReader = rasterio.open(s3_path)

        self._crs: str = str(self._dataset.crs)
        if self._crs != "EPSG:4326":
            self._dataset.close()
            raise ValueError(f"CogReader requires EPSG:4326 data, got {self._crs!r}")

        t = self._dataset.transform
        self._transform: tuple[float, ...] = (t.a, t.b, t.c, t.d, t.e, t.f)

        # Reject untiled (strip-layout) GeoTIFFs: block reads would
        # degenerate to row-level granularity, making format comparison
        # meaningless.  Use the GDAL tiled flag rather than block-size
        # heuristics, which miss small-image strips.
        if not self._dataset.profile.get("tiled", False):
            self._dataset.close()
            raise ValueError(f"CogReader requires a tiled GeoTIFF: {cog_path}")

        # Derive chunk size from the COG's internal tile structure
        block_shapes = self._dataset.block_shapes
        self._chunk_rows: int = block_shapes[0][0]
        self._chunk_cols: int = block_shapes[0][1]

        # Chunk cache: (chunk_row, chunk_col) -> chunk data
        self._cache: dict[tuple[int, int], NDArray[np.uint8]] = {}

    def close(self) -> None:
        """Close the underlying rasterio dataset."""
        self._dataset.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @property
    def crs(self) -> str:
        return self._crs

    @property
    def transform(self) -> tuple[float, ...]:
        return self._transform

    @property
    def shape(self) -> tuple[int, int]:
        return (self._dataset.height, self._dataset.width)

    @property
    def chunk_shape(self) -> tuple[int, int]:
        return (self._chunk_rows, self._chunk_cols)

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def get_chunk(self, chunk_row: int, chunk_col: int) -> NDArray[np.uint8]:
        """Return the chunk as a numpy array (cache-through).

        Edge chunks may be smaller than chunk_shape.
        """
        chunk_key = (chunk_row, chunk_col)
        if chunk_key not in self._cache:
            self._cache[chunk_key] = self._fetch_chunk(chunk_row, chunk_col)
        return self._cache[chunk_key]

    def _fetch_chunk(self, chunk_row: int, chunk_col: int) -> NDArray[np.uint8]:
        """Fetch chunk data directly from the rasterio dataset (no cache)."""
        rows, cols = self.shape
        r_start = chunk_row * self._chunk_rows
        c_start = chunk_col * self._chunk_cols
        r_end = min(r_start + self._chunk_rows, rows)
        c_end = min(c_start + self._chunk_cols, cols)
        window = Window(c_start, r_start, c_end - c_start, r_end - r_start)  # type: ignore[call-arg]
        data: NDArray[np.uint8] = self._dataset.read(1, window=window).astype(
            np.uint8, copy=False
        )
        return data

    def prefetch_chunks(self, keys: list[tuple[int, int]]) -> None:
        """Sequentially prefetch multiple chunks into the cache.

        Unlike ZarrReader, prefetch is not parallelized because
        rasterio's DatasetReader is not thread-safe.
        """
        for cr, cc in keys:
            if (cr, cc) not in self._cache:
                self._cache[(cr, cc)] = self._fetch_chunk(cr, cc)

    def get_cell(self, row: int, col: int) -> int:
        """Return the flow direction value of a cell (cache-through).

        Reads at chunk granularity (not 1x1 Window) for symmetry with
        ZarrReader: the chunk loaded during pour-point validation is
        reused as the BFS starting chunk.

        Raises:
            IndexError: If row or col is out of bounds.
        """
        rows, cols = self.shape
        if row < 0 or row >= rows or col < 0 or col >= cols:
            raise IndexError(
                f"Pixel coordinates ({row}, {col}) are out of bounds (shape: {rows}x{cols})"
            )

        chunk = self.get_chunk(row // self._chunk_rows, col // self._chunk_cols)
        return int(chunk[row % self._chunk_rows, col % self._chunk_cols])

    def latlon_to_pixel(self, latitude: float, longitude: float) -> tuple[int, int]:
        """Convert latitude/longitude to pixel coordinates (row, col).

        Raises:
            ValueError: If the coordinates are outside the data extent.
        """
        pixel_width = self._transform[0]
        origin_x = self._transform[2]
        pixel_height = abs(self._transform[4])
        origin_y = self._transform[5]

        row = math.floor((origin_y - latitude) / pixel_height)
        col = math.floor((longitude - origin_x) / pixel_width)

        rows, cols = self.shape
        if row < 0 or row >= rows or col < 0 or col >= cols:
            raise ValueError(
                f"Coordinates ({latitude}, {longitude}) are outside the data extent"
            )
        return row, col
