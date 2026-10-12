"""Object storage for the Legal Document Vault."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path

from app.config import get_settings

LOCAL_VAULT_DIR = Path(__file__).resolve().parent.parent / "data" / "vault"


class ObjectStorage(ABC):
    @abstractmethod
    def upload_object(self, key: str, data: bytes, content_type: str) -> None: ...

    @abstractmethod
    def get_download_url(self, key: str, document_id: str, expires_in: int = 3600) -> str: ...

    @abstractmethod
    def delete_object(self, key: str) -> None: ...

    @abstractmethod
    def read_object(self, key: str) -> bytes:
        """Only used server-side for text extraction/embedding right after upload — never to serve a download response to the client."""
        ...


class R2ObjectStorage(ObjectStorage):
    def __init__(self):
        import boto3

        settings = get_settings()
        self._bucket = settings.r2_bucket_name
        self._client = boto3.client(
            "s3",
            endpoint_url=settings.r2_endpoint_url,
            aws_access_key_id=settings.r2_access_key_id,
            aws_secret_access_key=settings.r2_secret_access_key,
            region_name="auto",
        )

    def upload_object(self, key: str, data: bytes, content_type: str) -> None:
        self._client.put_object(Bucket=self._bucket, Key=key, Body=data, ContentType=content_type)

    def get_download_url(self, key: str, document_id: str, expires_in: int = 3600) -> str:
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self._bucket, "Key": key},
            ExpiresIn=expires_in,
        )

    def delete_object(self, key: str) -> None:
        self._client.delete_object(Bucket=self._bucket, Key=key)

    def read_object(self, key: str) -> bytes:
        response = self._client.get_object(Bucket=self._bucket, Key=key)
        return response["Body"].read()


class LocalDiskObjectStorage(ObjectStorage):
    """Dev/test fallback used when R2 credentials aren't configured."""

    def __init__(self):
        LOCAL_VAULT_DIR.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:


        base = LOCAL_VAULT_DIR.resolve()
        path = (base / key).resolve()
        if not path.is_relative_to(base):
            raise ValueError("invalid object key")
        return path

    def upload_object(self, key: str, data: bytes, content_type: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def get_download_url(self, key: str, document_id: str, expires_in: int = 3600) -> str:


        return f"/api/vault/documents/local-download/{document_id}"

    def delete_object(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    def read_object(self, key: str) -> bytes:
        return self._path(key).read_bytes()


@lru_cache()
def get_object_storage() -> ObjectStorage:
    settings = get_settings()
    if settings.r2_endpoint_url and settings.r2_access_key_id and settings.r2_secret_access_key:
        return R2ObjectStorage()
    print("[ObjectStorage] R2 not configured — using local-disk fallback for the Vault.")
    return LocalDiskObjectStorage()
