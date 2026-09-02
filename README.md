# extract-watershed-api

A serverless API that extracts the upstream watershed of a clicked point on demand from nationwide flow direction data (J-FlwDir).

## Repository structure

Monorepo with multiple projects managed by a uv workspace.

| Directory | Description |
|---|---|
| `converter/` | CLI to convert J-FlwDir tiles → base GeoTIFF → COG / Zarr variants |
| `api/` | Extraction core (D8 reverse BFS, COG/Zarr readers) + Lambda service layer + Dockerfile |
| `benchmark/` | Measurement harnesses (local/S3: `run_local.py`, Lambda: `run_remote.py`) and result CSVs |
| `infra/` | AWS CDK (Python) stack: Docker Lambda + Function URL + output bucket |
| `demo/` | Leaflet web map demo (click to extract watershed) |

Inside `api/`, the core modules (`watershed.models` / `extractor` /
`readers` / `writers`) — also used by `benchmark` — are separated from
the Lambda-only service layer (`watershed.service`).  Dependencies flow
in one direction only: service layer → core.

## Setup

Requirements: [uv](https://docs.astral.sh/uv/), Node.js (for the CDK CLI),
Docker (for building the Lambda image).  GDAL CLI (gdalinfo) is used by
some tests but is optional (auto-skipped).

```bash
uv sync --all-packages
```

Note: `uv sync` alone does not install workspace member (api, converter,
etc.) dependencies.  **Always use `--all-packages`**.

## Data preparation pipeline

Input data is [J-FlwDir](https://global-hydrodynamics.github.io/J-FlwDir/)
(a surface flow direction map of Japan by the Yamazaki Lab at the
University of Tokyo, 1 arc-second ≈ 30 m).  Register for access, download
the flow direction (dir) tiles, and place them in `dir/`.  The data itself
is not included in this repository.

```bash
# 1° tiles → nationwide base GeoTIFF (ocean areas without tiles filled with 247 = ocean)
uv run jflwdir-convert mosaic dir data/dir_base.tif --pattern '*_dir.tif'

# Base GeoTIFF → COG (block size is an experimental variable)
uv run jflwdir-convert to-cog data/dir_base.tif data/dir_cog_b512.tif --block-size 512

# Base GeoTIFF → Zarr v3 (chunk/shard/compression level are experimental variables)
uv run jflwdir-convert to-zarr data/dir_base.tif data/dir_zarr_c512.zarr --chunk-size 512

# Batch-generate all benchmark variants (COG×4 block sizes, Zarr×4 chunk sizes, sharding×2)
bash benchmark/make_variants.sh
```

The generated Zarr follows the same metadata conventions as GDAL's Zarr
driver (`_CRS` / coordinate arrays X·Y / `fill_value`), so it can be
opened directly with gdalinfo or QGIS (sharded stores are not yet
supported as of GDAL 3.11).

J-FlwDir value definitions: 1,2,4,...,128 = D8 flow direction, 0 = outlet,
247 = ocean, 255 = inland depression.

## Local watershed extraction

```python
from watershed.extractor import extract_basin
from watershed.models import PourPoint
from watershed.readers import CogReader, ZarrReader

reader = ZarrReader("data/dir_zarr_c512.zarr")  # or CogReader / s3:// URL
result = extract_basin(reader, PourPoint(35.7446, 140.8507))  # Tone River outlet
print(result.cell_count, result.area_km2)  # 20,204,838 cells / ~15,555 km²
```

## API deployment and usage

```bash
cd infra
npx aws-cdk deploy   # Input bucket etc. configured via cdk.json context
```

Stack contents: Docker image Lambda (2048 MB / 120 s / reserved concurrency 10) +
Function URL (**no auth — pragmatic choice for benchmarking**) + output bucket
(presigned URL distribution, auto-deleted after 1 day).

```bash
curl -X POST "$FUNCTION_URL/extract-basin" -H 'Content-Type: application/json' \
  -d '{"latitude": 35.7446, "longitude": 140.8507, "format": "zarr"}'
```

The response includes results (cell count, area, bbox, presigned URLs for
GeoTIFF/PNG) as well as benchmarking information (per-phase timings,
bfs_mode, chunks fetched).  The `dataset` field can specify any variant
in the input bucket (allowing all variants to be measured without
redeploying).

Demo: open `demo/index.html`, set the API URL, and click on the map
(also configurable via `?api=<Function URL>` query parameter).

## Benchmarks

Measurement points are defined in `benchmark/scenarios.yaml` (with
expected cell counts; coordinates must use pixel center values — boundary
values can map to different pixels due to float differences between
readers).

```bash
# Local / local→S3 (1 measurement = 1 process to cold-start GDAL's in-process cache each time)
uv run --project benchmark python benchmark/run_local.py \
    --scenarios benchmark/scenarios.yaml --variants benchmark/variants_s3.yaml \
    --repeats 3 --label s3_cold

# Lambda (cold start forced via function config update; --memory-sizes enables memory sweep)
eval "$(aws configure export-credentials --profile <PROFILE> --format env)"
uv run --project benchmark python benchmark/run_remote.py \
    --scenarios benchmark/scenarios.yaml --variants benchmark/variants_s3.yaml \
    --warm-repeats 3 --label lambda_2048
```

Results are saved as CSV in `benchmark/results/` (only rows matching the
expected cell count and bfs_mode=chunked are marked valid).

## Development

```bash
uv run pytest converter/tests api/tests infra/tests   # tests (123 cases)
uv run ruff check . && uv run ruff format --check .   # lint / format
uv run mypy api/src converter/src benchmark/run_local.py benchmark/run_remote.py  # type check
```

CI (GitHub Actions) additionally verifies drift in `api/requirements.txt`
(fully pinned, exported from uv.lock for the Lambda image).  Regenerate
after updating uv.lock:

```bash
uv export --project api --no-dev --no-hashes --no-emit-project --no-emit-workspace \
  -o api/requirements.txt
```

## License

See [LICENSE](LICENSE).  Usage of J-FlwDir data is subject to the terms
of the data distributor.
