"""Data models and D8 flow direction encoding constants.

Defines data structures for watershed extraction (PourPoint / BoundingBox /
BasinResult), D8 flow direction code constants, and the FlowReader protocol.

J-FlwDir value definitions (official: https://global-hydrodynamics.github.io/J-FlwDir/):

- 1, 2, 4, 8, 16, 32, 64, 128 : D8 flow directions (clockwise from east)
- 0   : river mouth (0 in signed representation)
- 247 : ocean / undefined (-9 in signed representation)
- 255 : inland depression (-1 in signed representation)
"""

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

# BFS execution path ("chunked" = fast vectorized / "scalar" = fallback)
BfsMode = Literal["chunked", "scalar"]

# D8 flow direction code -> (drow, dcol) mapping
D8_OFFSETS: dict[int, tuple[int, int]] = {
    1: (0, 1),  # E
    2: (1, 1),  # SE
    4: (1, 0),  # S
    8: (1, -1),  # SW
    16: (0, -1),  # W
    32: (-1, -1),  # NW
    64: (-1, 0),  # N
    128: (-1, 1),  # NE
}

# Valid D8 flow direction codes
VALID_FLOW_CODES: frozenset[int] = frozenset(D8_OFFSETS.keys())

# Special values for non-flow cells (J-FlwDir official definition)
RIVER_MOUTH: int = 0  # river mouth
OCEAN: int = 247  # ocean / undefined
INLAND_DEPRESSION: int = 255  # inland depression

# Set of non-flow values (used for pour point validation and BFS exclusion)
NON_FLOW_VALUES: frozenset[int] = frozenset({RIVER_MOUTH, OCEAN, INLAND_DEPRESSION})

# Inflow reverse-lookup table: each entry is (drow, dcol, code).
# A neighbor at offset (drow, dcol) flows into the current cell
# when its flow direction equals code.
INFLOW_OFFSETS: list[tuple[int, int, int]] = [
    (-1, -1, 2),  # NW neighbor pointing SE(2) -> inflow
    (-1, 0, 4),  # N -> S(4)
    (-1, 1, 8),  # NE -> SW(8)
    (0, -1, 1),  # W -> E(1)
    (0, 1, 16),  # E -> W(16)
    (1, -1, 128),  # SW -> NE(128)
    (1, 0, 64),  # S -> N(64)
    (1, 1, 32),  # SE -> NW(32)
]


@dataclass(frozen=True)
class PourPoint:
    """Pour point for watershed extraction (WGS84 coordinates)."""

    latitude: float
    longitude: float


@dataclass(frozen=True)
class BoundingBox:
    """Bounding box with both pixel and geographic coordinates."""

    min_row: int
    max_row: int
    min_col: int
    max_col: int
    south: float
    north: float
    west: float
    east: float

    @property
    def height(self) -> int:
        return self.max_row - self.min_row + 1

    @property
    def width(self) -> int:
        return self.max_col - self.min_col + 1


@dataclass
class BasinResult:
    """Watershed extraction result."""

    mask: NDArray[np.uint8]
    bbox: BoundingBox
    cell_count: int
    area_km2: float
    crs: str
    transform: tuple[float, ...]
    pour_point: PourPoint
    bfs_mode: BfsMode


class FlowReader(Protocol):
    """Flow direction data reader interface.

    Both COG and Zarr readers implement this protocol, allowing the
    extraction algorithm to work independently of the storage format.
    """

    @property
    def crs(self) -> str: ...

    @property
    def transform(self) -> tuple[float, ...]: ...

    @property
    def shape(self) -> tuple[int, int]: ...

    def get_cell(self, row: int, col: int) -> int: ...

    def latlon_to_pixel(self, latitude: float, longitude: float) -> tuple[int, int]: ...
