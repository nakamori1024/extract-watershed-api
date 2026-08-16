"""CDK stack for the watershed extraction API.

Defines the full set of AWS resources: an output S3 bucket (auto-deleted
after 1 day), a Docker-image Lambda function with Function URL, and IAM
permissions.  Ported from the jflwdir-extract-api POC stack.

The Function URL uses AUTH_TYPE=NONE with CORS "*" as a pragmatic choice
for benchmarking.  Authentication must be added before exposing as a
public API.
"""

from pathlib import Path
from typing import Any

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_s3 as s3
from constructs import Construct

# Docker build context (api/ directory).  Resolved relative to this file
# so it does not depend on the working directory (repo_root/api).
_API_DIR = str(Path(__file__).resolve().parents[2] / "api")


class ApiStack(Stack):
    """Watershed extraction API stack."""

    def __init__(self, scope: Construct, construct_id: str, **kwargs: Any) -> None:
        super().__init__(scope, construct_id, **kwargs)

        input_bucket_name = self._require_context("inputBucketName")
        cog_s3_key = self._require_context("cogS3Key")
        zarr_s3_key = self._require_context("zarrS3Key")

        # --- Output S3 bucket (1-day lifecycle, auto-deleted on stack removal) ---
        output_bucket = s3.Bucket(
            self,
            "OutputBucket",
            lifecycle_rules=[
                s3.LifecycleRule(expiration=Duration.days(1)),
            ],
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        # --- Lambda function (Docker image) ---
        # reserved_concurrent_executions caps cost (prevents runaway during benchmarks)
        handler = lambda_.DockerImageFunction(
            self,
            "ExtractHandler",
            code=lambda_.DockerImageCode.from_image_asset(_API_DIR),
            architecture=lambda_.Architecture.X86_64,
            memory_size=2048,
            timeout=Duration.seconds(120),
            reserved_concurrent_executions=10,
            environment={
                "OUTPUT_BUCKET_NAME": output_bucket.bucket_name,
                "INPUT_BUCKET_NAME": input_bucket_name,
                "COG_S3_KEY": cog_s3_key,
                "ZARR_S3_KEY": zarr_s3_key,
            },
        )

        # --- Function URL (no auth, CORS enabled) ---
        fn_url = handler.add_function_url(
            auth_type=lambda_.FunctionUrlAuthType.NONE,
            cors=lambda_.FunctionUrlCorsOptions(
                allowed_origins=["*"],
                allowed_methods=[lambda_.HttpMethod.POST],
                allowed_headers=["Content-Type"],
                max_age=Duration.hours(1),
            ),
        )

        # --- Stack outputs ---
        CfnOutput(self, "FunctionUrl", value=fn_url.url)
        CfnOutput(self, "OutputBucketName", value=output_bucket.bucket_name)

        # --- IAM permissions ---
        output_bucket.grant_read_write(handler)

        # Input bucket is pre-existing (not managed by this stack).  Read-only access
        input_bucket = s3.Bucket.from_bucket_name(
            self, "InputBucket", input_bucket_name
        )
        input_bucket.grant_read(handler)

    def _require_context(self, key: str) -> str:
        """Retrieve a required CDK context value (raises on missing)."""
        value: str = self.node.try_get_context(key) or ""
        if not value:
            raise ValueError(
                f"CDK context '{key}' is required. Set it in cdk.json or "
                f"pass -c {key}=<value>"
            )
        return value
