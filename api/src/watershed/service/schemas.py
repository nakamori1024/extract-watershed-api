"""Pydantic request/response schemas for the watershed extraction API.

ExtractRequest validates coordinate range (Japan region) and format.
ExtractResponse includes benchmark information (per-phase timing, bfs_mode,
chunks fetched). Remote measurement is self-contained in the response,
eliminating the need for CloudWatch log collection.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ExtractRequest(BaseModel):
    """Request body for watershed extraction."""

    latitude: float = Field(
        ge=24.0, le=46.0, description="Latitude (EPSG:4326, decimal degrees)"
    )
    longitude: float = Field(
        ge=122.0, le=149.0, description="Longitude (EPSG:4326, decimal degrees)"
    )
    format: Literal["cog", "zarr"] = Field(description="Input data format")
    dataset: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9._/-]+$",
        description="S3 key in the input bucket (for benchmark variant selection. "
        "Falls back to the default key from environment variables if omitted)",
    )

    @field_validator("dataset")
    @classmethod
    def _validate_dataset_path(cls, v: str | None) -> str | None:
        """Reject keys that could lead to path traversal."""
        if v is not None and (v.startswith("/") or ".." in v):
            raise ValueError(f"Invalid dataset key: {v}")
        return v


class BBox(BaseModel):
    """Geographic bounding box (EPSG:4326, decimal degrees)."""

    north: float
    south: float
    east: float
    west: float


class ExtractResponse(BaseModel):
    """Response body for watershed extraction."""

    cell_count: int
    area_km2: float
    bbox: BBox
    geotiff_url: str
    png_url: str
    elapsed_sec: float
    format: Literal["cog", "zarr"]
    # Benchmark / verification fields
    bfs_mode: Literal["chunked", "scalar"]
    chunks_fetched: int
    phases: dict[str, float] = Field(
        description="Per-phase elapsed time (seconds): reader_init / extract / "
        "geotiff_write / geotiff_upload / png_write / png_upload"
    )


class ErrorResponse(BaseModel):
    """Error response body."""

    error: str
    message: str
