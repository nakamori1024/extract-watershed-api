"""Integration tests for pipeline.

Runs end-to-end from extraction to file output using real stores generated
by converter (only S3 upload is mocked).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import rasterio
from watershed.models import PourPoint
from watershed.service import pipeline

# Same values as the flow_tif fixture in conftest.py
ORIGIN_X = 139.0
ORIGIN_Y = 36.0
RES = 0.001


class TestRunPipeline:
    def test_end_to_end_with_zarr(self, zarr_store: str, tmp_path: Path) -> None:
        pour = PourPoint(ORIGIN_Y - (50 + 0.5) * RES, ORIGIN_X + (75 + 0.5) * RES)

        with (
            patch.object(pipeline, "upload_file", return_value="req-1/basin.tif"),
            patch.object(
                pipeline, "generate_presigned_url", return_value="https://signed"
            ),
        ):
            output = pipeline.run_pipeline(
                zarr_store, "zarr", pour, "req-1", tmp_dir=str(tmp_path)
            )

        # Phases are in execution order
        assert list(output.phases.keys()) == [
            "reader_init",
            "extract",
            "geotiff_write",
            "geotiff_upload",
            "png_write",
            "png_upload",
        ]
        assert all(v >= 0 for v in output.phases.values())

        # Intermediate files are actually written
        with rasterio.open(tmp_path / "basin.tif") as src:
            assert src.read(1).sum() == output.result.cell_count
        assert (tmp_path / "basin.png").exists()

        assert output.result.bfs_mode == "chunked"
        assert output.chunks_fetched > 0
        assert output.geotiff_url == "https://signed"

    def test_cog_reader_is_closed(self, cog_file: str, tmp_path: Path) -> None:
        """CogReader is closed even on error (non-flow cell)."""
        # (0,0) is a random field so it may not be a flow code. Find a sea-value cell.
        import numpy as np

        with rasterio.open(cog_file) as src:
            grid = src.read(1)
        rows, cols = np.where(grid == 247)
        row, col = int(rows[0]), int(cols[0])
        pour = PourPoint(ORIGIN_Y - (row + 0.5) * RES, ORIGIN_X + (col + 0.5) * RES)

        try:
            pipeline.run_pipeline(cog_file, "cog", pour, "req-2", tmp_dir=str(tmp_path))
        except ValueError as e:
            assert "non-flow" in str(e)
        else:
            raise AssertionError("ValueError should have been raised")
