"""Watershed extraction pipeline.

Executes reader init, extraction, GeoTIFF/PNG write, and S3 upload+presign
sequentially, returning per-phase elapsed times. Separated from the handler
(event parsing) so it can be unit-tested and run locally.
"""

import time
from dataclasses import dataclass
from typing import Literal

from watershed.extractor import extract_basin
from watershed.models import BasinResult, PourPoint
from watershed.readers import CogReader, ZarrReader
from watershed.service.storage import generate_presigned_url, upload_file
from watershed.writers import write_geotiff, write_png


@dataclass
class PipelineOutput:
    """Pipeline execution result."""

    result: BasinResult
    geotiff_url: str
    png_url: str
    chunks_fetched: int
    # Phase name -> elapsed time (seconds). Insertion order = execution order.
    phases: dict[str, float]


def run_pipeline(
    data_url: str,
    fmt: Literal["cog", "zarr"],
    pour_point: PourPoint,
    request_id: str,
    tmp_dir: str = "/tmp",
) -> PipelineOutput:
    """Run the full watershed extraction workflow.

    Args:
        data_url: Input flow direction data URL (s3:// or local path)
        fmt: Input data format
        pour_point: Starting point for watershed extraction
        request_id: Identifier used as S3 object key prefix
        tmp_dir: Output directory for intermediate files (/tmp on Lambda)

    Raises:
        ValueError: When the pour point is out of bounds or on a non-flow cell
            (propagated from extractor)
    """
    phases: dict[str, float] = {}
    tmp_tif = f"{tmp_dir}/basin.tif"
    tmp_png = f"{tmp_dir}/basin.png"

    def mark(name: str, since: float) -> float:
        now = time.perf_counter()
        phases[name] = round(now - since, 4)
        return now

    t = time.perf_counter()
    reader: CogReader | ZarrReader = (
        CogReader(data_url) if fmt == "cog" else ZarrReader(data_url)
    )
    t = mark("reader_init", t)

    try:
        result = extract_basin(reader, pour_point)
    finally:
        if isinstance(reader, CogReader):
            reader.close()
    chunks_fetched = reader.cache_size
    t = mark("extract", t)

    write_geotiff(result, tmp_tif)
    t = mark("geotiff_write", t)

    tif_key = upload_file(tmp_tif, request_id, "basin.tif")
    geotiff_url = generate_presigned_url(tif_key)
    t = mark("geotiff_upload", t)

    write_png(result, tmp_png)
    t = mark("png_write", t)

    png_key = upload_file(tmp_png, request_id, "basin.png")
    png_url = generate_presigned_url(png_key)
    mark("png_upload", t)

    return PipelineOutput(
        result=result,
        geotiff_url=geotiff_url,
        png_url=png_url,
        chunks_fetched=chunks_fetched,
        phases=phases,
    )
