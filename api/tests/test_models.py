"""Tests for models."""

from __future__ import annotations

from watershed.models import (
    D8_OFFSETS,
    INFLOW_OFFSETS,
    INLAND_DEPRESSION,
    NON_FLOW_VALUES,
    OCEAN,
    RIVER_MOUTH,
    VALID_FLOW_CODES,
    BoundingBox,
)


class TestD8Constants:
    def test_eight_directions(self) -> None:
        """D8 codes are 8 powers of two with no overlapping bits."""
        assert VALID_FLOW_CODES == {1, 2, 4, 8, 16, 32, 64, 128}

    def test_offsets_are_unit_neighbors(self) -> None:
        """Each direction points to one of the 8 neighbors (never itself)."""
        offsets = set(D8_OFFSETS.values())
        assert len(offsets) == 8
        for drow, dcol in offsets:
            assert (drow, dcol) != (0, 0)
            assert drow in (-1, 0, 1) and dcol in (-1, 0, 1)

    def test_inflow_offsets_are_inverse_of_d8(self) -> None:
        """Inflow table consistency: D8_OFFSETS[code] must equal (-drow, -dcol),
        i.e. the neighbor's flow direction points back to the current cell."""
        assert len(INFLOW_OFFSETS) == 8
        for drow, dcol, code in INFLOW_OFFSETS:
            assert D8_OFFSETS[code] == (-drow, -dcol)

    def test_non_flow_values_follow_official_definition(self) -> None:
        """J-FlwDir official: 0=river mouth, 247=ocean, 255=inland depression."""
        assert RIVER_MOUTH == 0
        assert OCEAN == 247
        assert INLAND_DEPRESSION == 255
        assert NON_FLOW_VALUES == {0, 247, 255}

    def test_non_flow_and_flow_codes_are_disjoint(self) -> None:
        assert NON_FLOW_VALUES.isdisjoint(VALID_FLOW_CODES)


class TestBoundingBox:
    def test_height_and_width_are_inclusive(self) -> None:
        """min/max are both inclusive, so height/width = max - min + 1."""
        bbox = BoundingBox(
            min_row=10,
            max_row=19,
            min_col=100,
            max_col=104,
            south=35.0,
            north=36.0,
            west=139.0,
            east=140.0,
        )
        assert bbox.height == 10
        assert bbox.width == 5

    def test_single_cell(self) -> None:
        bbox = BoundingBox(
            min_row=5,
            max_row=5,
            min_col=7,
            max_col=7,
            south=35.0,
            north=35.0,
            west=139.0,
            east=139.0,
        )
        assert bbox.height == 1
        assert bbox.width == 1
