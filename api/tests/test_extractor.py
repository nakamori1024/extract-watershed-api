"""Tests for extractor.

Uses the scalar BFS as a reference implementation and verifies that the
chunked (wavefront-vectorized) BFS produces identical results.  Mock
readers expose an ndarray through the FlowReader interface.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray
from watershed.extractor import extract_basin
from watershed.models import OCEAN, PourPoint

ORIGIN_X = 139.0
ORIGIN_Y = 36.0
RES = 0.001


class ScalarReader:
    """Exposes an ndarray as a FlowReader (triggers scalar BFS)."""

    def __init__(self, grid: NDArray[np.uint8]) -> None:
        self._grid = grid

    @property
    def crs(self) -> str:
        return "EPSG:4326"

    @property
    def transform(self) -> tuple[float, ...]:
        return (RES, 0.0, ORIGIN_X, 0.0, -RES, ORIGIN_Y)

    @property
    def shape(self) -> tuple[int, int]:
        return self._grid.shape  # type: ignore[return-value]

    def get_cell(self, row: int, col: int) -> int:
        return int(self._grid[row, col])

    def latlon_to_pixel(self, latitude: float, longitude: float) -> tuple[int, int]:
        row = math.floor((ORIGIN_Y - latitude) / RES)
        col = math.floor((longitude - ORIGIN_X) / RES)
        return row, col


class ChunkedReader(ScalarReader):
    """Reader with chunk access (triggers chunked BFS)."""

    def __init__(
        self, grid: NDArray[np.uint8], chunk_shape: tuple[int, int] = (8, 8)
    ) -> None:
        super().__init__(grid)
        self._chunk_shape = chunk_shape

    @property
    def chunk_shape(self) -> tuple[int, int]:
        return self._chunk_shape

    def get_chunk(self, chunk_row: int, chunk_col: int) -> NDArray[np.uint8]:
        h, w = self._chunk_shape
        return self._grid[
            chunk_row * h : (chunk_row + 1) * h, chunk_col * w : (chunk_col + 1) * w
        ]


class PrefetchRecordingReader(ChunkedReader):
    """Reader that records prefetch_chunks calls."""

    def __init__(
        self, grid: NDArray[np.uint8], chunk_shape: tuple[int, int] = (8, 8)
    ) -> None:
        super().__init__(grid, chunk_shape)
        self.prefetch_calls: list[list[tuple[int, int]]] = []

    def prefetch_chunks(self, keys: list[tuple[int, int]]) -> None:
        self.prefetch_calls.append(keys)


def pixel_to_latlon(row: int, col: int) -> tuple[float, float]:
    """Return the lat/lon of a pixel center (for specifying test pour points)."""
    return ORIGIN_Y - (row + 0.5) * RES, ORIGIN_X + (col + 0.5) * RES


def make_straight_river(length: int = 5, size: int = 9) -> NDArray[np.uint8]:
    """A straight river: cells in row size//2, cols 2..2+length-1, flowing east (1)."""
    grid = np.full((size, size), OCEAN, dtype=np.uint8)
    row = size // 2
    grid[row, 2 : 2 + length] = 1  # east
    return grid


class TestScalarBfs:
    def test_straight_river(self) -> None:
        """Straight east-flowing river: the entire upstream is extracted from the most downstream cell."""
        grid = make_straight_river(length=5, size=9)
        row = 9 // 2
        lat, lon = pixel_to_latlon(row, 6)  # most downstream (easternmost) cell

        result = extract_basin(ScalarReader(grid), PourPoint(lat, lon))

        assert result.cell_count == 5
        assert result.bbox.min_row == row and result.bbox.max_row == row
        assert result.bbox.min_col == 2 and result.bbox.max_col == 6
        assert result.mask.shape == (1, 5)
        assert (result.mask == 1).all()

    def test_pour_point_midstream_excludes_downstream(self) -> None:
        """Starting midstream excludes downstream cells."""
        grid = make_straight_river(length=5, size=9)
        row = 9 // 2
        lat, lon = pixel_to_latlon(row, 4)  # midstream

        result = extract_basin(ScalarReader(grid), PourPoint(lat, lon))

        assert result.cell_count == 3  # col 2, 3, 4
        assert result.bbox.max_col == 4

    def test_y_shaped_confluence(self) -> None:
        """Y-shaped confluence: two tributaries flow into a junction cell."""
        grid = np.full((9, 9), OCEAN, dtype=np.uint8)
        # Main: (4,4) flows east. NW(3,3) flows SE(2), SW(5,3) flows NE(128) into (4,4)
        grid[4, 4] = 1
        grid[3, 3] = 2  # SE -> flows into (4,4)
        grid[5, 3] = 128  # NE -> flows into (4,4)
        lat, lon = pixel_to_latlon(4, 4)

        result = extract_basin(ScalarReader(grid), PourPoint(lat, lon))

        assert result.cell_count == 3
        assert result.bbox.min_row == 3 and result.bbox.max_row == 5


class TestChunkedEquivalence:
    """Chunked BFS produces results identical to scalar BFS regardless of chunk size."""

    @pytest.mark.parametrize("chunk_shape", [(4, 4), (8, 8), (7, 5), (64, 64)])
    def test_random_flow_field(self, chunk_shape: tuple[int, int]) -> None:
        """Equivalence on a random flow field (covers chunk boundary crossings and edge chunks)."""
        rng = np.random.default_rng(42)
        grid = rng.choice(
            [1, 2, 4, 8, 16, 32, 64, 128, 0, 247, 255], size=(40, 50)
        ).astype(np.uint8)
        # Pick a cell with a valid flow code as the pour point
        row, col = 20, 25
        grid[row, col] = 1
        lat, lon = pixel_to_latlon(row, col)

        expected = extract_basin(ScalarReader(grid), PourPoint(lat, lon))
        actual = extract_basin(ChunkedReader(grid, chunk_shape), PourPoint(lat, lon))

        assert actual.cell_count == expected.cell_count
        assert actual.bbox == expected.bbox
        np.testing.assert_array_equal(actual.mask, expected.mask)
        assert actual.area_km2 == pytest.approx(expected.area_km2)

    def test_river_crossing_chunk_boundary(self) -> None:
        """A straight river crossing chunk boundaries is traced without gaps."""
        grid = np.full((20, 20), OCEAN, dtype=np.uint8)
        grid[10, 2:18] = 1  # 16 east-flowing cells (spans 3 chunks at chunk_size=8)
        lat, lon = pixel_to_latlon(10, 17)

        result = extract_basin(ChunkedReader(grid, (8, 8)), PourPoint(lat, lon))

        assert result.cell_count == 16
        assert result.bbox.min_col == 2 and result.bbox.max_col == 17

    def test_prefetch_is_called(self) -> None:
        """prefetch_chunks is called when extraction spans multiple chunks."""
        grid = np.full((20, 20), OCEAN, dtype=np.uint8)
        grid[10, 2:18] = 1
        lat, lon = pixel_to_latlon(10, 17)
        reader = PrefetchRecordingReader(grid, (8, 8))

        extract_basin(reader, PourPoint(lat, lon))

        assert reader.prefetch_calls  # called at least once
        requested = {key for call in reader.prefetch_calls for key in call}
        assert len(requested) >= 2  # multiple chunks requested


class TestBfsModeObservability:
    """The BFS execution path is observable from the result."""

    def test_chunked_reader_uses_chunked_path(self) -> None:
        grid = make_straight_river()
        lat, lon = pixel_to_latlon(9 // 2, 6)
        result = extract_basin(ChunkedReader(grid), PourPoint(lat, lon))
        assert result.bfs_mode == "chunked"

    def test_scalar_reader_falls_back_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        grid = make_straight_river()
        lat, lon = pixel_to_latlon(9 // 2, 6)
        with caplog.at_level("WARNING", logger="watershed.extractor"):
            result = extract_basin(ScalarReader(grid), PourPoint(lat, lon))
        assert result.bfs_mode == "scalar"
        assert any("scalar BFS" in message for message in caplog.messages)


class TestGeoValues:
    def test_bbox_geographic_coordinates(self) -> None:
        """bbox geographic coordinates refer to pixel boundaries (outer edges)."""
        grid = make_straight_river(length=5, size=9)
        row = 9 // 2
        lat, lon = pixel_to_latlon(row, 6)

        result = extract_basin(ScalarReader(grid), PourPoint(lat, lon))

        assert result.bbox.north == pytest.approx(ORIGIN_Y - row * RES)
        assert result.bbox.south == pytest.approx(ORIGIN_Y - (row + 1) * RES)
        assert result.bbox.west == pytest.approx(ORIGIN_X + 2 * RES)
        assert result.bbox.east == pytest.approx(ORIGIN_X + 7 * RES)

    def test_clipped_transform_origin(self) -> None:
        """The clipped transform origin is the northwest corner of the bbox."""
        grid = make_straight_river(length=5, size=9)
        row = 9 // 2
        lat, lon = pixel_to_latlon(row, 6)

        result = extract_basin(ScalarReader(grid), PourPoint(lat, lon))

        a, b, c, d, e, f = result.transform
        assert (a, b, d, e) == (RES, 0.0, 0.0, -RES)
        assert c == pytest.approx(ORIGIN_X + 2 * RES)
        assert f == pytest.approx(ORIGIN_Y - row * RES)

    def test_area_uses_cos_latitude_correction(self) -> None:
        """Area is corrected by cos(center latitude) in the longitudinal direction (matches hand calculation)."""
        grid = np.full((9, 9), OCEAN, dtype=np.uint8)
        row, col = 4, 4
        grid[row, col] = 1  # single-cell basin with no upstream
        lat, lon = pixel_to_latlon(row, col)

        result = extract_basin(ScalarReader(grid), PourPoint(lat, lon))

        lat_center = ORIGIN_Y - (row + 0.5) * RES
        expected = (
            (
                RES * 111.32 * math.cos(math.radians(lat_center))
            )  # longitudinal (cos-corrected)
            * (RES * 111.32)  # latitudinal
        )
        assert result.cell_count == 1
        assert result.area_km2 == pytest.approx(expected, rel=1e-12)
        # Clearly differs from the uncorrected value (~0.81x at 36°N)
        assert result.area_km2 < (RES * 111.32) ** 2 * 0.85


class TestErrorHandling:
    def test_pour_point_on_non_flow_cell_raises(self) -> None:
        grid = np.full((9, 9), OCEAN, dtype=np.uint8)
        lat, lon = pixel_to_latlon(4, 4)
        with pytest.raises(ValueError, match="non-flow cell"):
            extract_basin(ScalarReader(grid), PourPoint(lat, lon))

    @pytest.mark.parametrize("value", [0, 247, 255])
    def test_all_non_flow_values_rejected(self, value: int) -> None:
        grid = np.full((9, 9), 1, dtype=np.uint8)
        grid[4, 4] = value
        lat, lon = pixel_to_latlon(4, 4)
        with pytest.raises(ValueError, match="non-flow cell"):
            extract_basin(ScalarReader(grid), PourPoint(lat, lon))

    def test_pour_point_out_of_bounds_raises(self) -> None:
        """Out-of-bounds pour points are rejected before negative indices can wrap around."""
        grid = np.full((9, 9), 1, dtype=np.uint8)
        with pytest.raises(ValueError, match="outside the data extent"):
            extract_basin(
                ScalarReader(grid), PourPoint(37.0, 139.004)
            )  # north of extent
        with pytest.raises(ValueError, match="outside the data extent"):
            extract_basin(
                ScalarReader(grid), PourPoint(35.9955, 138.0)
            )  # west of extent
