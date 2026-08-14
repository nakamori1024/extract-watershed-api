#!/usr/bin/env bash
# Batch-generate benchmark variants from a base GeoTIFF.
#
# Test matrix:
# - Granularity comparison: COG and Zarr at matching sizes (256 / 512 / 1024 / 2048)
# - Zarr-specific: sharded stores (small chunk x large shard), 2 configurations
# - Compression fixed to each format's default (COG=LZW, Zarr=zstd level 0)
#
# See benchmark/README.md for usage.
set -euo pipefail

BASE="data/dir_base.tif"
OUT="${1:-data}"

run() {
    echo "=== $* ==="
    uv run jflwdir-convert "$@"
}

# COG: block size variants
for b in 256 512 1024 2048; do
    run to-cog "$BASE" "$OUT/dir_cog_b${b}.tif" --block-size "$b" --overwrite
done

# Zarr: chunk size variants (no sharding)
for c in 256 512 1024 2048; do
    run to-zarr "$BASE" "$OUT/dir_zarr_c${c}.zarr" --chunk-size "$c" --overwrite
done

# Zarr: sharded (read unit = small chunk, S3 object = large shard)
run to-zarr "$BASE" "$OUT/dir_zarr_c256_s2048.zarr" --chunk-size 256 --shard-size 2048 --overwrite
run to-zarr "$BASE" "$OUT/dir_zarr_c512_s2048.zarr" --chunk-size 512 --shard-size 2048 --overwrite

echo "=== done ==="
du -sh "$OUT"/dir_cog_b*.tif "$OUT"/dir_zarr_*.zarr
