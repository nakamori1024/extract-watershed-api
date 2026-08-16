"""Remote (Lambda) benchmark harness.

Sends "variant x scenario" requests to a deployed Function URL and saves
per-phase timings from the response into a CSV file.

- Cold measurement: Lambda execution environments cannot be discarded
  directly, so a dummy environment variable (BENCH_EPOCH) is changed to
  invalidate existing warm environments; the first request right after
  is recorded as cold.
- Warm measurement: after the cold request, the same request is sent
  N more times.
- Memory sweep: when --memory-sizes is given, memory is changed in order
  and the same matrix is measured at each level (Lambda vCPU scales with
  memory, useful for profiling CPU-bound bottlenecks).
- Validation is the same as the local harness (cell_count match +
  bfs_mode=chunked).

Usage (AWS credentials required for management API):
    eval "$(aws configure export-credentials --profile nakamori --format env)"
    uv run --project benchmark python benchmark/run_remote.py \\
        --scenarios benchmark/scenarios.yaml \\
        --variants benchmark/variants_s3.yaml \\
        --warm-repeats 3 --label lambda_2048
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
import httpx
import yaml

DEFAULT_STACK = "ExtractWatershedApiStack"

RESULT_FIELDS = [
    "timestamp",
    "label",
    "memory_mb",
    "variant",
    "format",
    "dataset",
    "scenario",
    "cold",
    "run",
    "client_total_s",
    "elapsed_sec",
    "reader_init_s",
    "extract_s",
    "geotiff_write_s",
    "geotiff_upload_s",
    "png_write_s",
    "png_upload_s",
    "chunks_fetched",
    "cell_count",
    "bfs_mode",
    "valid",
    "note",
]


def resolve_stack(stack_name: str) -> tuple[str, str]:
    """Retrieve the Function URL and Lambda function name from a CloudFormation stack."""
    cfn = boto3.client("cloudformation")
    outputs = cfn.describe_stacks(StackName=stack_name)["Stacks"][0]["Outputs"]
    fn_url = next(o["OutputValue"] for o in outputs if o["OutputKey"] == "FunctionUrl")

    resources = cfn.list_stack_resources(StackName=stack_name)["StackResourceSummaries"]
    # The stack also contains a custom-resource Lambda for auto_delete_objects,
    # so filter by logical ID to identify the extraction handler
    fn_name = next(
        r["PhysicalResourceId"]
        for r in resources
        if r["ResourceType"] == "AWS::Lambda::Function"
        and r["LogicalResourceId"].startswith("ExtractHandler")
    )
    return fn_url.rstrip("/"), fn_name


class LambdaController:
    """Controls Lambda configuration changes (force cold start, change memory)."""

    def __init__(self, function_name: str) -> None:
        self._client = boto3.client("lambda")
        self._function_name = function_name

    def _wait_updated(self, timeout_s: float = 120.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            conf = self._client.get_function_configuration(
                FunctionName=self._function_name
            )
            if conf["LastUpdateStatus"] == "Successful":
                return
            if conf["LastUpdateStatus"] == "Failed":
                raise RuntimeError(f"Function update failed: {conf}")
            time.sleep(1)
        raise TimeoutError("Timed out waiting for function update to complete")

    def _current_env(self) -> dict[str, str]:
        conf = self._client.get_function_configuration(FunctionName=self._function_name)
        return dict(conf.get("Environment", {}).get("Variables", {}))

    def force_cold(self) -> None:
        """Overwrite a dummy env var to invalidate existing warm environments."""
        env = self._current_env()
        env["BENCH_EPOCH"] = str(uuid.uuid4())
        self._client.update_function_configuration(
            FunctionName=self._function_name, Environment={"Variables": env}
        )
        self._wait_updated()

    def set_memory(self, memory_mb: int) -> None:
        """Change memory setting (this also invalidates existing environments)."""
        self._client.update_function_configuration(
            FunctionName=self._function_name, MemorySize=memory_mb
        )
        self._wait_updated()

    def current_memory(self) -> int:
        conf = self._client.get_function_configuration(FunctionName=self._function_name)
        return int(conf["MemorySize"])


def dataset_key(uri: str) -> str:
    """Extract the key portion from an s3://bucket/key URI."""
    return uri.split("/", 3)[3]


def invoke_once(
    client: httpx.Client,
    fn_url: str,
    variant: dict[str, str],
    scenario: dict[str, Any],
    *,
    cold: bool,
    run_idx: int,
    label: str,
    memory_mb: int,
) -> dict[str, Any]:
    """Send a single request and return a measurement row."""
    row: dict[str, Any] = {
        "timestamp": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
        "label": label,
        "memory_mb": memory_mb,
        "variant": variant["name"],
        "format": variant["format"],
        "dataset": dataset_key(variant["uri"]),
        "scenario": scenario["id"],
        "cold": cold,
        "run": run_idx,
        "valid": False,
        "note": "",
    }

    payload = {
        "latitude": scenario["latitude"],
        "longitude": scenario["longitude"],
        "format": variant["format"],
        "dataset": row["dataset"],
    }

    t0 = time.perf_counter()
    try:
        resp = client.post(f"{fn_url}/extract-basin", json=payload)
    except httpx.HTTPError as e:
        row["note"] = f"HTTP error: {e}"
        return row
    row["client_total_s"] = round(time.perf_counter() - t0, 4)

    if resp.status_code != 200:
        row["note"] = f"status={resp.status_code}: {resp.text[:200]}"
        return row

    body = resp.json()
    phases = body.get("phases", {})
    row.update(
        elapsed_sec=body.get("elapsed_sec"),
        reader_init_s=phases.get("reader_init"),
        extract_s=phases.get("extract"),
        geotiff_write_s=phases.get("geotiff_write"),
        geotiff_upload_s=phases.get("geotiff_upload"),
        png_write_s=phases.get("png_write"),
        png_upload_s=phases.get("png_upload"),
        chunks_fetched=body.get("chunks_fetched"),
        cell_count=body.get("cell_count"),
        bfs_mode=body.get("bfs_mode"),
    )

    expected = scenario.get("expected_cell_count")
    if body.get("bfs_mode") != "chunked":
        row["note"] = "bfs_mode is not chunked"
    elif expected is not None and body.get("cell_count") != expected:
        row["note"] = f"cell count mismatch: expected={expected}"
    else:
        row["valid"] = True
    return row


def summarize(rows: list[dict[str, Any]]) -> str:
    """Per-condition median summary of extract time (cold/warm split)."""
    lines = [
        (
            f"{'memory':>6} {'variant':<16} {'scenario':<15} {'cold':<5} {'n':>2} "
            f"{'extract(s)':>10} {'total(s)':>9}"
        )
    ]
    keys = sorted(
        {(r["memory_mb"], r["variant"], r["scenario"], r["cold"]) for r in rows}
    )
    for memory_mb, variant, scenario, cold in keys:
        group = [
            r
            for r in rows
            if (r["memory_mb"], r["variant"], r["scenario"], r["cold"])
            == (memory_mb, variant, scenario, cold)
            and r["valid"]
        ]
        if not group:
            lines.append(
                f"{memory_mb:>6} {variant:<16} {scenario:<15} {cold!s:<5} no valid measurements"
            )
            continue
        ext = statistics.median(float(r["extract_s"]) for r in group)
        tot = statistics.median(float(r["elapsed_sec"]) for r in group)
        lines.append(
            f"{memory_mb:>6} {variant:<16} {scenario:<15} {cold!s:<5} "
            f"{len(group):>2} {ext:>10.2f} {tot:>9.2f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lambda remote benchmark")
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--variants", required=True)
    parser.add_argument("--stack-name", default=DEFAULT_STACK)
    parser.add_argument("--warm-repeats", type=int, default=3)
    parser.add_argument("--no-cold", action="store_true", help="Skip cold measurement")
    parser.add_argument(
        "--memory-sizes",
        nargs="*",
        type=int,
        help="Memory sweep (MB). Uses current setting if omitted",
    )
    parser.add_argument("--scenario-ids", nargs="*")
    parser.add_argument("--variant-names", nargs="*")
    parser.add_argument("--out-dir", default="benchmark/results")
    parser.add_argument("--label", default="lambda")
    args = parser.parse_args()

    scenarios = yaml.safe_load(Path(args.scenarios).read_text())["scenarios"]
    variants = yaml.safe_load(Path(args.variants).read_text())["variants"]
    if args.scenario_ids:
        scenarios = [s for s in scenarios if s["id"] in args.scenario_ids]
    if args.variant_names:
        variants = [v for v in variants if v["name"] in args.variant_names]

    fn_url, fn_name = resolve_stack(args.stack_name)
    controller = LambdaController(fn_name)
    original_memory = controller.current_memory()
    memory_sizes = args.memory_sizes or [original_memory]
    print(f"Function: {fn_name} (current {original_memory}MB) / URL: {fn_url}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    now_local = datetime.now(UTC).astimezone()
    out_path = out_dir / f"{now_local.strftime('%Y%m%d_%H%M%S')}_{args.label}.csv"

    runs_per_pair = (0 if args.no_cold else 1) + args.warm_repeats
    total = len(memory_sizes) * len(variants) * len(scenarios) * runs_per_pair
    done = 0
    rows: list[dict[str, Any]] = []

    try:
        with (
            out_path.open("w", newline="") as f,
            # Timeout accounts for cold starts (measured over 40s at 2048MB)
            httpx.Client(timeout=180.0) as client,
        ):
            writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
            writer.writeheader()
            for memory_mb in memory_sizes:
                if args.memory_sizes:
                    print(f"=== Changing memory to {memory_mb}MB ===")
                    controller.set_memory(memory_mb)
                for variant in variants:
                    for scenario in scenarios:
                        if not args.no_cold:
                            controller.force_cold()
                        for run_idx in range(1, runs_per_pair + 1):
                            cold = (not args.no_cold) and run_idx == 1
                            row = invoke_once(
                                client,
                                fn_url,
                                variant,
                                scenario,
                                cold=cold,
                                run_idx=run_idx,
                                label=args.label,
                                memory_mb=memory_mb,
                            )
                            rows.append(row)
                            writer.writerow(row)
                            f.flush()
                            done += 1
                            status = "ok" if row["valid"] else f"NG ({row['note']})"
                            print(
                                f"[{done}/{total}] {memory_mb}MB "
                                f"{variant['name']} x {scenario['id']} "
                                f"{'cold' if cold else 'warm'}{run_idx}: "
                                f"{row.get('elapsed_sec', '-')}s {status}"
                            )
    finally:
        # Restore original memory setting after a memory sweep
        if args.memory_sizes and controller.current_memory() != original_memory:
            print(f"Restoring memory to {original_memory}MB")
            controller.set_memory(original_memory)

    print(f"\nResults: {out_path}")
    print(summarize(rows))


if __name__ == "__main__":
    main()
