"""Local benchmark harness.

Runs every combination of variant (COG/Zarr configuration) x scenario
(pour point) x repeat, and saves per-phase timing results to a CSV file.

- A fresh reader is created for each run, so the chunk cache always
  starts cold.  (OS page cache may still be warm for local files,
  causing run 1 to differ from later runs — use the run column to
  distinguish them during analysis.)
- By default each measurement runs in a separate spawned process.
  GDAL's VSIcurl cache and connection pools are shared process-wide,
  so repeating runs in the same process warms them from run 2 onward.
  This causes library-specific caching behaviour to masquerade as a
  format difference (observed: COG run 2+ is ~5x faster in-process).
  Use --no-isolate to run in-process for warm-cache reference values.
- A row is marked valid=true only when the cell count matches the
  expected value in scenarios.yaml AND bfs_mode is "chunked".
  Invalid measurements should be excluded from aggregated statistics.

Usage:
    uv run --project benchmark python benchmark/run_local.py \\
        --scenarios benchmark/scenarios.yaml \\
        --variants benchmark/variants.yaml \\
        --repeats 5 --label local_disk
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from watershed.extractor import extract_basin
from watershed.models import PourPoint
from watershed.readers import CogReader, ZarrReader

RESULT_FIELDS = [
    "timestamp",
    "label",
    "variant",
    "format",
    "uri",
    "scenario",
    "run",
    "reader_init_s",
    "extract_s",
    "total_s",
    "chunks_fetched",
    "cell_count",
    "area_km2",
    "bfs_mode",
    "valid",
    "note",
]


def make_reader(fmt: str, uri: str) -> CogReader | ZarrReader:
    if fmt == "cog":
        return CogReader(uri)
    if fmt == "zarr":
        return ZarrReader(uri)
    raise ValueError(f"Unknown format: {fmt}")


def run_once(
    variant: dict[str, str], scenario: dict[str, Any], run_idx: int, label: str
) -> dict[str, Any]:
    """Run a single measurement from reader creation through extraction."""
    row: dict[str, Any] = {
        "timestamp": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
        "label": label,
        "variant": variant["name"],
        "format": variant["format"],
        "uri": variant["uri"],
        "scenario": scenario["id"],
        "run": run_idx,
        "valid": False,
        "note": "",
    }

    t0 = time.perf_counter()
    reader = make_reader(variant["format"], variant["uri"])
    t1 = time.perf_counter()
    try:
        result = extract_basin(
            reader, PourPoint(scenario["latitude"], scenario["longitude"])
        )
    except ValueError as e:
        row["note"] = f"extraction error: {e}"
        return row
    finally:
        if isinstance(reader, CogReader):
            reader.close()
    t2 = time.perf_counter()

    row.update(
        reader_init_s=round(t1 - t0, 4),
        extract_s=round(t2 - t1, 4),
        total_s=round(t2 - t0, 4),
        chunks_fetched=reader.cache_size,
        cell_count=result.cell_count,
        area_km2=round(result.area_km2, 1),
        bfs_mode=result.bfs_mode,
    )

    # Validity check — keep invalid measurements out of aggregated stats
    expected = scenario.get("expected_cell_count")
    if result.bfs_mode != "chunked":
        row["note"] = "bfs_mode is not chunked"
    elif expected is not None and result.cell_count != expected:
        row["note"] = f"cell count mismatch: expected={expected}"
    else:
        row["valid"] = True
    return row


def summarize(rows: list[dict[str, Any]]) -> str:
    """Format a median summary table grouped by variant and scenario."""
    lines = [
        (
            f"{'variant':<14} {'scenario':<16} {'n':>2} "
            f"{'init(s)':>8} {'extract(s)':>10} {'chunks':>7}"
        )
    ]
    keys = sorted({(r["variant"], r["scenario"]) for r in rows})
    for variant, scenario in keys:
        group = [
            r for r in rows if (r["variant"], r["scenario"]) == (variant, scenario)
        ]
        valid = [r for r in group if r["valid"]]
        if not valid:
            lines.append(f"{variant:<14} {scenario:<16} all {len(group)} runs invalid")
            continue
        init_med = statistics.median(r["reader_init_s"] for r in valid)
        ext_med = statistics.median(r["extract_s"] for r in valid)
        chunks = valid[0]["chunks_fetched"]
        lines.append(
            f"{variant:<14} {scenario:<16} {len(valid):>2} "
            f"{init_med:>8.3f} {ext_med:>10.3f} {chunks:>7}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local benchmark harness")
    parser.add_argument("--scenarios", required=True, help="path to scenarios.yaml")
    parser.add_argument("--variants", required=True, help="path to variants.yaml")
    parser.add_argument("--repeats", type=int, default=5, help="number of repetitions")
    parser.add_argument(
        "--out-dir",
        default="benchmark/results",
        help="output directory for result CSVs",
    )
    parser.add_argument("--label", default="local", help="label recorded in CSV")
    parser.add_argument(
        "--scenario-ids",
        nargs="*",
        help="scenario ids to run (default: all)",
    )
    parser.add_argument(
        "--no-isolate",
        action="store_true",
        help="run in-process instead of spawning (warm-cache reference values)",
    )
    args = parser.parse_args()

    scenarios = yaml.safe_load(Path(args.scenarios).read_text())["scenarios"]
    variants = yaml.safe_load(Path(args.variants).read_text())["variants"]
    if args.scenario_ids:
        scenarios = [s for s in scenarios if s["id"] in args.scenario_ids]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    now_local = datetime.now(UTC).astimezone()
    out_path = out_dir / f"{now_local.strftime('%Y%m%d_%H%M%S')}_{args.label}.csv"

    # Process isolation: max_tasks_per_child=1 destroys the worker after
    # each measurement, clearing GDAL and other process-wide caches.
    executor: ProcessPoolExecutor | None = None
    if not args.no_isolate:
        executor = ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
            max_tasks_per_child=1,
        )

    rows: list[dict[str, Any]] = []
    total = len(variants) * len(scenarios) * args.repeats
    done = 0
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for variant in variants:
            for scenario in scenarios:
                for run_idx in range(1, args.repeats + 1):
                    if executor is not None:
                        row = executor.submit(
                            run_once, variant, scenario, run_idx, args.label
                        ).result()
                    else:
                        row = run_once(variant, scenario, run_idx, args.label)
                    rows.append(row)
                    writer.writerow(row)
                    f.flush()  # preserve results even if interrupted
                    done += 1
                    status = "ok" if row["valid"] else f"NG ({row['note']})"
                    print(
                        f"[{done}/{total}] {variant['name']} x {scenario['id']} "
                        f"run{run_idx}: {row.get('total_s', '-')}s {status}"
                    )

    if executor is not None:
        executor.shutdown()

    print(f"\nResults: {out_path}")
    print(summarize(rows))


if __name__ == "__main__":
    main()
