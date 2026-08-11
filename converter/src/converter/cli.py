"""CLI definition for the jflwdir-convert command."""

from __future__ import annotations

import time

import click

from converter.mosaic import mosaic_tiles
from converter.to_cog import convert_to_cog
from converter.to_zarr import convert_geotiff_to_zarr


@click.group()
def main() -> None:
    """Format conversion tool for J-FlwDir flow direction data."""


@main.command("mosaic")
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("output_path", type=click.Path())
@click.option(
    "--pattern",
    default="*.tif",
    show_default=True,
    help="Filename pattern for tiles (e.g. '*_dir.tif').",
)
@click.option(
    "--fill",
    default=247,
    show_default=True,
    help="Fill value for areas with no tile coverage (247 = ocean in J-FlwDir).",
)
@click.option(
    "--nodata",
    default=None,
    type=int,
    help="NoData tag value for the output. Omit to write no tag (matches original data).",
)
@click.option(
    "--overwrite", is_flag=True, help="Overwrite the output if it already exists."
)
def mosaic(
    input_dir: str,
    output_path: str,
    pattern: str,
    fill: int,
    nodata: int | None,
    overwrite: bool,
) -> None:
    """Mosaic J-FlwDir 1-degree tiles into a single GeoTIFF."""
    start = time.perf_counter()
    height, width, n_tiles = mosaic_tiles(
        input_dir,
        output_path,
        pattern=pattern,
        fill=fill,
        nodata=nodata,
        overwrite=overwrite,
    )
    elapsed = time.perf_counter() - start
    click.echo(f"Mosaic complete: {output_path}")
    click.echo(f"  tiles: {n_tiles}, size: {width}x{height}, elapsed: {elapsed:.1f}s")


@main.command("to-cog")
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_path", type=click.Path())
@click.option(
    "--block-size",
    default=512,
    show_default=True,
    help="Internal tile side length in pixels (must be a multiple of 16).",
)
@click.option(
    "--compress",
    default="LZW",
    show_default=True,
    type=click.Choice(["LZW", "DEFLATE", "ZSTD", "NONE"], case_sensitive=False),
    help="Compression method.",
)
@click.option(
    "--overviews/--no-overviews",
    default=True,
    show_default=True,
    help="Generate overviews (downsampling uses NEAREST).",
)
@click.option(
    "--overwrite", is_flag=True, help="Overwrite the output if it already exists."
)
def to_cog(
    input_path: str,
    output_path: str,
    block_size: int,
    compress: str,
    overviews: bool,
    overwrite: bool,
) -> None:
    """Convert a GeoTIFF to COG format (using GDAL's COG driver)."""
    start = time.perf_counter()
    convert_to_cog(
        input_path,
        output_path,
        block_size=block_size,
        compress=compress.upper(),
        overviews=overviews,
        overwrite=overwrite,
    )
    elapsed = time.perf_counter() - start
    click.echo(f"Conversion complete: {output_path}")
    click.echo(
        f"  block size: {block_size}, compress: {compress}, elapsed: {elapsed:.1f}s"
    )


@main.command("to-zarr")
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_path", type=click.Path())
@click.option(
    "--chunk-size",
    default=512,
    show_default=True,
    help="Chunk side length in pixels.",
)
@click.option(
    "--shard-size",
    default=None,
    type=int,
    help="Shard side length in pixels (must be a multiple of chunk-size). Omit to disable sharding.",
)
@click.option(
    "--zstd-level",
    default=0,
    show_default=True,
    help="zstd compression level.",
)
@click.option(
    "--fill-value",
    default=247,
    show_default=True,
    help="fill_value for unwritten chunks. Chunks where all pixels equal this value are omitted, "
    "so setting 247 (ocean) for J-FlwDir skips ocean-only chunks.",
)
@click.option(
    "--overwrite", is_flag=True, help="Overwrite the output if it already exists."
)
def to_zarr(
    input_path: str,
    output_path: str,
    chunk_size: int,
    shard_size: int | None,
    zstd_level: int,
    fill_value: int,
    overwrite: bool,
) -> None:
    """Convert a GeoTIFF to a Zarr v3 store."""
    start = time.perf_counter()
    z = convert_geotiff_to_zarr(
        input_path,
        output_path,
        chunk_size=chunk_size,
        shard_size=shard_size,
        zstd_level=zstd_level,
        fill_value=fill_value,
        overwrite=overwrite,
    )
    elapsed = time.perf_counter() - start

    click.echo(f"Conversion complete: {output_path}")
    click.echo(f"  shape: {z.shape}, chunks: {z.chunks}, shards: {z.shards}")
    click.echo(f"  zstd level: {zstd_level}, elapsed: {elapsed:.1f}s")
