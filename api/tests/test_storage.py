"""Tests for storage (S3 client is mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from watershed.service import storage


@pytest.fixture(autouse=True)
def output_bucket_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OUTPUT_BUCKET_NAME", "test-bucket")


class TestUploadFile:
    def test_uploads_and_returns_key(self) -> None:
        mock_client = MagicMock()
        with patch.object(storage, "_s3_client", mock_client):
            key = storage.upload_file("/tmp/basin.tif", "req-123", "basin.tif")

        assert key == "req-123/basin.tif"
        mock_client.upload_file.assert_called_once_with(
            "/tmp/basin.tif", "test-bucket", "req-123/basin.tif"
        )

    def test_missing_bucket_env_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OUTPUT_BUCKET_NAME")
        with pytest.raises(RuntimeError, match="OUTPUT_BUCKET_NAME"):
            storage.upload_file("/tmp/basin.tif", "req-123", "basin.tif")


class TestGeneratePresignedUrl:
    def test_presigned_url_params(self) -> None:
        mock_client = MagicMock()
        mock_client.generate_presigned_url.return_value = "https://example/signed"
        with patch.object(storage, "_s3_client", mock_client):
            url = storage.generate_presigned_url("req-123/basin.png")

        assert url == "https://example/signed"
        mock_client.generate_presigned_url.assert_called_once_with(
            "get_object",
            Params={"Bucket": "test-bucket", "Key": "req-123/basin.png"},
            ExpiresIn=3600,
        )
