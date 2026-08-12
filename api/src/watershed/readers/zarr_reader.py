"""Zarr store flow direction data reader (S3-capable, with chunk cache).

Reads the GDAL-convention store layout produced by converter (to-zarr):

- The group contains a main 2-D array and coordinate arrays ``X`` / ``Y``
- CRS is stored as WKT in the main array's ``_CRS`` attribute
- Transform is reconstructed from the first two elements of the ``X`` / ``Y``
  coordinate arrays (pixel-center values)

Cell access is cached at chunk granularity; repeated access to the same
chunk is served from cache.  prefetch_chunks uses a ThreadPoolExecutor
to fetch multiple chunks in parallel, hiding S3 access latency.
"""

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import zarr
import zarr.storage
from numpy.typing import NDArray
from rasterio.crs import CRS


class ZarrReader:
    """FlowReader implementation that reads flow direction data from a Zarr store.

    Args:
        zarr_url: Local path or S3 URL (s3://bucket/key.zarr).
    """

    def __init__(self, zarr_url: str) -> None:
        if not zarr_url.startswith("s3://"):
            if not Path(zarr_url).exists():
                raise FileNotFoundError(f"Zarr store not found: {zarr_url}")
            store: str | zarr.storage.FsspecStore = zarr_url
        else:
            # S3 credentials are resolved automatically (e.g. Lambda execution role)
            store = zarr.storage.FsspecStore.from_url(zarr_url, read_only=True)

        group = zarr.open_group(store=store, mode="r")

        # Coordinate arrays X / Y (contract with converter)
        try:
            x_arr = group["X"]
            y_arr = group["Y"]
        except KeyError as e:
            raise ValueError(
                f"Coordinate arrays X / Y not found (may not be a "
                f"converter-generated store): {zarr_url}"
            ) from e
        assert isinstance(x_arr, zarr.Array) and isinstance(y_arr, zarr.Array)

        # Main array: exactly one array besides X / Y
        mains = [arr for name, arr in group.arrays() if name not in ("X", "Y")]
        if len(mains) != 1:
            raise ValueError(
                f"Expected exactly one main array, found {len(mains)}: {zarr_url}"
            )
        self._array: zarr.Array = mains[0]

        # CRS: normalize from _CRS attribute WKT to EPSG string
        crs_attr = self._array.attrs.get("_CRS")
        if not isinstance(crs_attr, dict) or "wkt" not in crs_attr:
            raise ValueError(f"_CRS attribute (wkt) not found: {zarr_url}")
        crs_obj = CRS.from_wkt(str(crs_attr["wkt"]))
        epsg = crs_obj.to_epsg()
        self._crs: str = f"EPSG:{epsg}" if epsg is not None else crs_obj.to_wkt()

        # Transform: reconstruct from the first two pixel-center values of X / Y
        if x_arr.shape[0] < 2 or y_arr.shape[0] < 2:
            raise ValueError(
                f"Coordinate arrays too short (need at least 2 elements): {zarr_url}"
            )
        x0, x1 = (float(v) for v in np.asarray(x_arr[0:2]))
        y0, y1 = (float(v) for v in np.asarray(y_arr[0:2]))
        xres = x1 - x0
        yres = y0 - y1  # Y is in descending order (north to south)
        self._transform: tuple[float, ...] = (
            xres,
            0.0,
            x0 - xres / 2,
            0.0,
            -yres,
            y0 + yres / 2,
        )

        # Chunk size.  For sharded stores, uses the inner chunk
        # (= the partial-read unit)
        chunks = self._array.chunks
        self._chunk_rows: int = int(chunks[0])
        self._chunk_cols: int = int(chunks[1])

        # Chunk cache: (chunk_row, chunk_col) -> chunk data
        self._cache: dict[tuple[int, int], NDArray[np.uint8]] = {}

    @property
    def crs(self) -> str:
        return self._crs

    @property
    def transform(self) -> tuple[float, ...]:
        return self._transform

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self._array.shape[0]), int(self._array.shape[1]))

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
        """Fetch chunk data directly from the Zarr array (no cache)."""
        rows, cols = self.shape
        r_start = chunk_row * self._chunk_rows
        c_start = chunk_col * self._chunk_cols
        r_end = min(r_start + self._chunk_rows, rows)
        c_end = min(c_start + self._chunk_cols, cols)
        return np.asarray(
            self._array[r_start:r_end, c_start:c_end],
            dtype=np.uint8,
        )

    def prefetch_chunks(self, keys: list[tuple[int, int]]) -> None:
        """Prefetch multiple chunks in parallel into the cache.

        Already-cached chunks are skipped.  Uses ThreadPoolExecutor
        to improve throughput for S3 access.
        """
        missing = [k for k in keys if k not in self._cache]
        if not missing:
            return

        with ThreadPoolExecutor(max_workers=min(len(missing), 8)) as pool:
            futures = {
                pool.submit(self._fetch_chunk, cr, cc): (cr, cc) for cr, cc in missing
            }
            for future in as_completed(futures):
                self._cache[futures[future]] = future.result()

    def get_cell(self, row: int, col: int) -> int:
        """Return the flow direction value of a cell (cache-through).

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
