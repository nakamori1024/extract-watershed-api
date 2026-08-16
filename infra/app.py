#!/usr/bin/env python3
"""CDK app entry point."""

import os

import aws_cdk as cdk
from infra.api_stack import ApiStack

app = cdk.App()
ApiStack(
    app,
    "ExtractWatershedApiStack",
    # Deploy to the account and region from the current CLI profile
    env=cdk.Environment(
        account=os.getenv("CDK_DEFAULT_ACCOUNT"),
        region=os.getenv("CDK_DEFAULT_REGION"),
    ),
)

app.synth()
