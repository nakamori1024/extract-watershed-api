"""Watershed extraction via chunk-aware BFS.

Traces upstream cells from a pour point using reverse BFS on D8 flow
direction data, and returns the result as a BasinResult with a binary
mask clipped to a bounding box.

When the reader supports chunk access (get_chunk / chunk_shape), a fast
wavefront-based path using vectorized NumPy operations is used.
Otherwise, a cell-by-cell scalar BFS is used as a fallback.  The scalar
implementation also serves as a reference for equivalence testing.
"""

import logging
from collections import deque

import numpy as np
from numpy.typing import NDArray

from watershed.models import (
    INFLOW_OFFSETS,
    NON_FLOW_VALUES,
    BasinResult,
    BfsMode,
    BoundingBox,
    FlowReader,
    PourPoint,
)

logger = logging.getLogger(__name__)

# Approximate distance per degree of latitude/longitude (km), for spherical area calculation
KM_PER_DEGREE = 111.32

# ------------------------------------------------------------------
# Scalar BFS (fallback for readers that only implement FlowReader)
# ------------------------------------------------------------------


def _bfs_scalar(
    reader: FlowReader,
    start_row: int,
    start_col: int,
    total_rows: int,
    total_cols: int,
) -> set[tuple[int, int]]:
    """Naive cell-by-cell BFS."""
    visited: set[tuple[int, int]] = set()
    queue: deque[tuple[int, int]] = deque()
    queue.append((start_row, start_col))

    while queue:
        r, c = queue.popleft()
        if (r, c) in visited:
            continue
        visited.add((r, c))

        for dr, dc, expected_code in INFLOW_OFFSETS:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= total_rows or nc < 0 or nc >= total_cols:
                continue
            if (nr, nc) in visited:
                continue
            if reader.get_cell(nr, nc) == expected_code:
                queue.append((nr, nc))

    return visited


def _result_from_visited_set(
    visited: set[tuple[int, int]],
) -> tuple[int, int, int, int, int, NDArray[np.uint8]]:
    """Compute bounding box and mask from a visited cell set."""
    min_row = min(r for r, _ in visited)
    max_row = max(r for r, _ in visited)
    min_col = min(c for _, c in visited)
    max_col = max(c for _, c in visited)

    height = max_row - min_row + 1
    width = max_col - min_col + 1
    mask = np.zeros((height, width), dtype=np.uint8)
    for r, c in visited:
        mask[r - min_row, c - min_col] = 1

    return min_row, max_row, min_col, max_col, len(visited), mask


# ------------------------------------------------------------------
# Chunked vectorized BFS (fast path)
# ------------------------------------------------------------------


def _bfs_chunked(
    reader: FlowReader,
    start_row: int,
    start_col: int,
    total_rows: int,
    total_cols: int,
    chunk_h: int,
    chunk_w: int,
) -> dict[tuple[int, int], NDArray[np.bool_]]:
    """Wavefront-based fast BFS.

    The frontier is maintained as numpy arrays of global coordinates,
    and neighbor candidates in all 8 directions are evaluated in bulk
    per chunk.

    Returns:
        visited_chunks: (chunk_row, chunk_col) -> boolean visited array
    """
    get_chunk = reader.get_chunk  # type: ignore[attr-defined]

    visited_chunks: dict[tuple[int, int], NDArray[np.bool_]] = {}
    flow_cache: dict[tuple[int, int], NDArray[np.uint8]] = {}

    def get_visited(cr: int, cc: int) -> NDArray[np.bool_]:
        if (cr, cc) not in visited_chunks:
            # Edge chunks may be smaller than the full chunk size
            actual_h = min(chunk_h, total_rows - cr * chunk_h)
            actual_w = min(chunk_w, total_cols - cc * chunk_w)
            visited_chunks[(cr, cc)] = np.zeros((actual_h, actual_w), dtype=np.bool_)
        return visited_chunks[(cr, cc)]

    def get_flow(cr: int, cc: int) -> NDArray[np.uint8]:
        if (cr, cc) not in flow_cache:
            flow_cache[(cr, cc)] = get_chunk(cr, cc)
        return flow_cache[(cr, cc)]

    # Mark the start cell as visited
    start_cr = start_row // chunk_h
    start_cc = start_col // chunk_w
    get_visited(start_cr, start_cc)[start_row % chunk_h, start_col % chunk_w] = True

    # Frontier: arrays of global coordinates
    frontier_rows = np.array([start_row], dtype=np.int32)
    frontier_cols = np.array([start_col], dtype=np.int32)

    # Convert INFLOW_OFFSETS to numpy arrays once outside the loop
    offsets_dr = np.array([o[0] for o in INFLOW_OFFSETS], dtype=np.int32)
    offsets_dc = np.array([o[1] for o in INFLOW_OFFSETS], dtype=np.int32)
    offsets_code = np.array([o[2] for o in INFLOW_OFFSETS], dtype=np.uint8)

    n_chunk_cols = (total_cols + chunk_w - 1) // chunk_w

    while len(frontier_rows) > 0:
        n = len(frontier_rows)

        # Generate all 8-direction candidates at once (length 8*n)
        cand_rows = np.repeat(frontier_rows, 8) + np.tile(offsets_dr, n)
        cand_cols = np.repeat(frontier_cols, 8) + np.tile(offsets_dc, n)
        expected = np.tile(offsets_code, n)

        # Bounds check
        valid = (
            (cand_rows >= 0)
            & (cand_rows < total_rows)
            & (cand_cols >= 0)
            & (cand_cols < total_cols)
        )
        cand_rows = cand_rows[valid]
        cand_cols = cand_cols[valid]
        expected = expected[valid]

        if len(cand_rows) == 0:
            break

        # Chunk indices and chunk-local coordinates
        cand_cr = cand_rows // chunk_h
        cand_cc = cand_cols // chunk_w
        cand_lr = (cand_rows % chunk_h).astype(np.intp)
        cand_lc = (cand_cols % chunk_w).astype(np.intp)

        # Sort by chunk key and group candidates
        chunk_keys = cand_cr * n_chunk_cols + cand_cc
        sort_idx = np.argsort(chunk_keys, kind="mergesort")
        sorted_keys = chunk_keys[sort_idx]

        # Detect group boundaries
        diffs = np.where(np.diff(sorted_keys))[0] + 1
        starts = np.empty(len(diffs) + 1, dtype=np.intp)
        ends = np.empty(len(diffs) + 1, dtype=np.intp)
        starts[0] = 0
        starts[1:] = diffs
        ends[:-1] = diffs
        ends[-1] = len(sorted_keys)

        # Prefetch all chunks needed for this wavefront
        if hasattr(reader, "prefetch_chunks"):
            needed_keys: list[tuple[int, int]] = []
            for i in range(len(starts)):
                idx_first = sort_idx[starts[i]]
                cr_val = int(cand_cr[idx_first])
                cc_val = int(cand_cc[idx_first])
                if (cr_val, cc_val) not in flow_cache:
                    needed_keys.append((cr_val, cc_val))
            if needed_keys:
                reader.prefetch_chunks(needed_keys)  # type: ignore[attr-defined]

        new_rows_list: list[NDArray[np.int32]] = []
        new_cols_list: list[NDArray[np.int32]] = []

        for i in range(len(starts)):
            idx = sort_idx[starts[i] : ends[i]]

            cr_val = int(cand_cr[idx[0]])
            cc_val = int(cand_cc[idx[0]])
            lr = cand_lr[idx]
            lc = cand_lc[idx]
            exp = expected[idx]

            flow = get_flow(cr_val, cc_val)
            vis = get_visited(cr_val, cc_val)

            # Accept only cells whose flow code matches and are not yet visited
            hit = (flow[lr, lc] == exp) & ~vis[lr, lc]
            if not np.any(hit):
                continue

            new_lr = lr[hit]
            new_lc = lc[hit]

            # Deduplicate cells within the same chunk
            if len(new_lr) > 1:
                local_keys = new_lr * chunk_w + new_lc
                _, unique_idx = np.unique(local_keys, return_index=True)
                new_lr = new_lr[unique_idx]
                new_lc = new_lc[unique_idx]

            vis[new_lr, new_lc] = True
            new_rows_list.append((cr_val * chunk_h + new_lr).astype(np.int32))
            new_cols_list.append((cc_val * chunk_w + new_lc).astype(np.int32))

        if new_rows_list:
            frontier_rows = np.concatenate(new_rows_list)
            frontier_cols = np.concatenate(new_cols_list)
        else:
            break

    return visited_chunks


def _result_from_visited_chunks(
    visited_chunks: dict[tuple[int, int], NDArray[np.bool_]],
    chunk_h: int,
    chunk_w: int,
) -> tuple[int, int, int, int, int, NDArray[np.uint8]]:
    """Compute bounding box and mask directly from visited_chunks."""
    global_min_row = np.iinfo(np.int32).max
    global_max_row = 0
    global_min_col = np.iinfo(np.int32).max
    global_max_col = 0
    cell_count = 0

    # Accumulate bounding box and cell count
    for (cr, cc), vis_arr in visited_chunks.items():
        if not np.any(vis_arr):
            continue
        local_rows, local_cols = np.where(vis_arr)
        g_rows = local_rows + cr * chunk_h
        g_cols = local_cols + cc * chunk_w
        global_min_row = min(global_min_row, int(g_rows.min()))
        global_max_row = max(global_max_row, int(g_rows.max()))
        global_min_col = min(global_min_col, int(g_cols.min()))
        global_max_col = max(global_max_col, int(g_cols.max()))
        cell_count += len(local_rows)

    min_row = int(global_min_row)
    max_row = int(global_max_row)
    min_col = int(global_min_col)
    max_col = int(global_max_col)

    # Build the mask
    mask = np.zeros((max_row - min_row + 1, max_col - min_col + 1), dtype=np.uint8)
    for (cr, cc), vis_arr in visited_chunks.items():
        if not np.any(vis_arr):
            continue
        local_rows, local_cols = np.where(vis_arr)
        g_rows = local_rows + cr * chunk_h
        g_cols = local_cols + cc * chunk_w
        mask[g_rows - min_row, g_cols - min_col] = 1

    return min_row, max_row, min_col, max_col, cell_count, mask


# ------------------------------------------------------------------
# Area calculation
# ------------------------------------------------------------------


def _area_km2(
    mask: NDArray[np.uint8],
    min_row: int,
    pixel_width: float,
    pixel_height: float,
    origin_y: float,
) -> float:
    """Compute watershed area in km² from the mask.

    The longitudinal cell width shrinks with latitude by cos(lat),
    so each row is scaled by the cosine of its center latitude.
    """
    row_counts = mask.sum(axis=1, dtype=np.int64)
    lat_centers = origin_y - (min_row + np.arange(mask.shape[0]) + 0.5) * pixel_height
    cell_width_km = pixel_width * KM_PER_DEGREE * np.cos(np.radians(lat_centers))
    cell_height_km = pixel_height * KM_PER_DEGREE
    return float(np.sum(row_counts * cell_width_km) * cell_height_km)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def extract_basin(reader: FlowReader, pour_point: PourPoint) -> BasinResult:
    """Extract the upstream watershed from a pour point.

    Args:
        reader: Flow direction data reader (FlowReader compatible).
        pour_point: Starting point for watershed extraction.

    Raises:
        ValueError: The pour point is outside the data extent, or falls
            on a non-flow cell (river mouth, ocean, or inland depression).
    """
    total_rows, total_cols = reader.shape

    # 1. Convert coordinates to pixel indices and verify they are in bounds
    #    (negative indices would wrap around in numpy, so they must be rejected)
    row, col = reader.latlon_to_pixel(pour_point.latitude, pour_point.longitude)
    if not (0 <= row < total_rows and 0 <= col < total_cols):
        raise ValueError(
            f"Coordinates ({pour_point.latitude}, {pour_point.longitude}) "
            f"are outside the data extent"
        )

    # 2. Verify the pour point cell has a valid flow direction
    cell_value = reader.get_cell(row, col)
    if cell_value in NON_FLOW_VALUES:
        raise ValueError(
            f"Coordinates ({pour_point.latitude}, {pour_point.longitude}) "
            f"fall on a non-flow cell (value={cell_value})"
        )

    # 3. Run BFS to find upstream cells.
    #    Dispatch is via duck typing; bfs_mode is recorded in the result
    #    so that benchmarks can detect unintended fallback to scalar BFS.
    bfs_mode: BfsMode
    if hasattr(reader, "get_chunk") and hasattr(reader, "chunk_shape"):
        bfs_mode = "chunked"
        chunk_h, chunk_w = reader.chunk_shape  # type: ignore[attr-defined]
        visited_chunks = _bfs_chunked(
            reader, row, col, total_rows, total_cols, chunk_h, chunk_w
        )
        min_row, max_row, min_col, max_col, cell_count, mask = (
            _result_from_visited_chunks(visited_chunks, chunk_h, chunk_w)
        )
    else:
        bfs_mode = "scalar"
        logger.warning(
            "Reader %s does not support chunk access; falling back to scalar BFS "
            "(slow for large basins)",
            type(reader).__name__,
        )
        visited = _bfs_scalar(reader, row, col, total_rows, total_cols)
        min_row, max_row, min_col, max_col, cell_count, mask = _result_from_visited_set(
            visited
        )

    # 4. Build the BasinResult
    pixel_width = reader.transform[0]
    origin_x = reader.transform[2]
    pixel_height = abs(reader.transform[4])
    origin_y = reader.transform[5]

    bbox = BoundingBox(
        min_row=min_row,
        max_row=max_row,
        min_col=min_col,
        max_col=max_col,
        north=origin_y - min_row * pixel_height,
        south=origin_y - (max_row + 1) * pixel_height,
        west=origin_x + min_col * pixel_width,
        east=origin_x + (max_col + 1) * pixel_width,
    )

    area_km2 = _area_km2(mask, min_row, pixel_width, pixel_height, origin_y)

    clipped_transform = (
        pixel_width,
        0.0,
        origin_x + min_col * pixel_width,
        0.0,
        -pixel_height,
        origin_y - min_row * pixel_height,
    )

    return BasinResult(
        mask=mask,
        bbox=bbox,
        cell_count=cell_count,
        area_km2=area_km2,
        crs=reader.crs,
        transform=clipped_transform,
        pour_point=pour_point,
        bfs_mode=bfs_mode,
    )
