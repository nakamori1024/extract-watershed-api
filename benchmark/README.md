# benchmark

Benchmark harness for quantitative comparison of COG vs Zarr read performance.

## File layout

| File | Description |
|---|---|
| `run_local.py` | Benchmark runner (variant x scenario x repeat → CSV) |
| `scenarios.yaml` | Pour-point catalogue (coordinates and expected cell counts) |
| `variants.yaml` | Variant definitions for local data |
| `variants_s3.yaml` | Variant definitions for S3 data |
| `make_variants.sh` | Batch-generate benchmark variants from `dir_base.tif` |

## Generating variants

Data variants (different chunk/block sizes) can be batch-generated with
`make_variants.sh`.

```bash
bash benchmark/make_variants.sh          # output to data/
bash benchmark/make_variants.sh /tmp/out  # custom output directory
```

Requires `data/dir_base.tif` (J-FlwDir flow direction GeoTIFF) as source data.

## Running locally

```bash
uv run --project benchmark python benchmark/run_local.py \
    --scenarios benchmark/scenarios.yaml \
    --variants benchmark/variants.yaml \
    --repeats 5 --label local_disk
```

Results are saved as CSV files in `benchmark/results/`.

### Options

| Option | Default | Description |
|---|---|---|
| `--scenarios` | (required) | Path to scenarios.yaml |
| `--variants` | (required) | Path to variants.yaml |
| `--repeats` | 5 | Number of repetitions |
| `--out-dir` | `benchmark/results` | Output directory for result CSVs |
| `--label` | `local` | Label recorded in CSV |
| `--scenario-ids` | all scenarios | Scenario IDs to run |
| `--no-isolate` | (off) | Disable process isolation (see below) |

### Process isolation

By default, each measurement runs in a separately spawned process.
GDAL's VSIcurl cache and connection pools are shared process-wide, so
repeating runs in the same process warms them from run 2 onward, causing
library-specific caching behaviour to appear as a format difference
(observed: COG run 2+ is ~5x faster in-process).

Use `--no-isolate` to disable process isolation and collect warm-cache
reference values.

## Running with S3 data

When using S3 variants, export temporary SSO credentials as environment
variables before running.

```bash
# 1. SSO login (browser auth, only needed on first run or session expiry)
aws sso login --profile <your-profile>

# 2. Export temporary credentials and run the benchmark
eval "$(aws configure export-credentials --profile <your-profile> --format env)"
uv run --project benchmark python benchmark/run_local.py \
    --scenarios benchmark/scenarios.yaml \
    --variants benchmark/variants_s3.yaml \
    --repeats 5 --label s3_from_local
```

The credentials exported by `eval` (`AWS_ACCESS_KEY_ID` /
`AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN`) exist only within that
shell process — nothing is written to configuration files. They are
gone when the shell is closed and become invalid when the SSO session
expires.

### Why environment variables?

An alternative is to pass `AWS_PROFILE=<name>` and let each library
resolve the SSO profile directly, but SSO profile support varies across
libraries (botocore fully supports it; GDAL support is version-dependent).
The three-variable set is the lowest common denominator that every AWS
client implementation reliably reads, ensuring that authentication path
differences between COG (GDAL) and Zarr (s3fs/botocore) do not leak
into benchmark results.
