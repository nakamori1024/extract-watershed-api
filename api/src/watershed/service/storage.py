"""Upload output files to S3 and generate presigned URLs.

Upload watershed extraction output (GeoTIFF / PNG) to the output bucket and
return a presigned URL valid for 1 hour. Object key format:
{request_id}/basin.tif|png

Bucket name is retrieved from the OUTPUT_BUCKET_NAME environment variable
(injected into Lambda by the CDK stack).
"""

import os

import boto3

_PRESIGNED_URL_EXPIRY = 3600  # 1 hour

# Created at module load time (= Lambda INIT phase) to avoid including
# client creation cost in request processing time measurements.
_s3_client = boto3.client("s3")


def _get_output_bucket() -> str:
    """Return the output bucket name from the OUTPUT_BUCKET_NAME environment variable."""
    bucket = os.environ.get("OUTPUT_BUCKET_NAME")
    if not bucket:
        raise RuntimeError("OUTPUT_BUCKET_NAME environment variable is not set")
    return bucket


def upload_file(local_path: str, request_id: str, filename: str) -> str:
    """Upload a local file to the S3 output bucket.

    Args:
        local_path: Local file path (e.g., /tmp/basin.tif)
        request_id: Request identifier (UUID)
        filename: Output filename (e.g., basin.tif, basin.png)

    Returns:
        S3 object key (e.g., {request_id}/basin.tif)
    """
    bucket = _get_output_bucket()
    key = f"{request_id}/{filename}"
    _s3_client.upload_file(local_path, bucket, key)
    return key


def generate_presigned_url(key: str) -> str:
    """Generate a presigned URL (valid for 1 hour) for an S3 object.

    Args:
        key: S3 object key (e.g., {request_id}/basin.tif)
    """
    url: str = _s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": _get_output_bucket(), "Key": key},
        ExpiresIn=_PRESIGNED_URL_EXPIRY,
    )
    return url
