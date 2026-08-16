"""Synth tests for ApiStack.

Template.from_stack does not build Docker images (assets only compute a
directory fingerprint), so template resource definitions can be verified
quickly.
"""

import aws_cdk as cdk
import pytest
from aws_cdk.assertions import Match, Template
from infra.api_stack import ApiStack

CONTEXT = {
    "inputBucketName": "test-input-bucket",
    "cogS3Key": "bench/dir_cog.tif",
    "zarrS3Key": "bench/dir.zarr",
}


@pytest.fixture(scope="module")
def template() -> Template:
    app = cdk.App(context=CONTEXT)
    stack = ApiStack(app, "TestStack")
    return Template.from_stack(stack)


class TestApiStack:
    def test_output_bucket_lifecycle_one_day(self, template: Template) -> None:
        template.has_resource_properties(
            "AWS::S3::Bucket",
            {
                "LifecycleConfiguration": {
                    "Rules": [Match.object_like({"ExpirationInDays": 1})]
                },
            },
        )

    def test_lambda_configuration(self, template: Template) -> None:
        template.has_resource_properties(
            "AWS::Lambda::Function",
            Match.object_like(
                {
                    "MemorySize": 2048,
                    "Timeout": 120,
                    "PackageType": "Image",
                    "ReservedConcurrentExecutions": 10,
                    "Environment": {
                        "Variables": Match.object_like(
                            {
                                "INPUT_BUCKET_NAME": "test-input-bucket",
                                "COG_S3_KEY": "bench/dir_cog.tif",
                                "ZARR_S3_KEY": "bench/dir.zarr",
                            }
                        )
                    },
                }
            ),
        )

    def test_function_url_without_auth(self, template: Template) -> None:
        template.has_resource_properties(
            "AWS::Lambda::Url",
            Match.object_like({"AuthType": "NONE"}),
        )

    def test_missing_context_raises(self) -> None:
        app = cdk.App()
        with pytest.raises(ValueError, match="inputBucketName"):
            ApiStack(app, "MissingContextStack")
