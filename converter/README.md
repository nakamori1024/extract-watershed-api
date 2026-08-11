# converter

Convert [J-FlwDir](https://www.csis.u-tokyo.ac.jp/~yamana/j-flwdir/) flow direction tiles to COG and Zarr v3 formats.

## Pipeline

```
J-FlwDir 1-deg tiles  -->  mosaic  -->  GeoTIFF  -->  to-cog  -->  COG
                                                  -->  to-zarr -->  Zarr v3
```

## Setup

```bash
cd converter
uv sync
```

## Usage

The CLI is installed as `jflwdir-convert`. Run `--help` on any command for full option details.

```bash
# 1. Mosaic tiles into a single GeoTIFF
jflwdir-convert mosaic ./tiles mosaic.tif --pattern "*_dir.tif"

# 2. Convert to Cloud Optimized GeoTIFF
jflwdir-convert to-cog mosaic.tif output.cog.tif

# 3. Convert to Zarr v3
jflwdir-convert to-zarr mosaic.tif output.zarr --fill-value 247
```

## Testing

```bash
uv run pytest
```
