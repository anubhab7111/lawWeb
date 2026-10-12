"""Shared streaming-read guard for file uploads."""

from app.deps.errors import MessageHTTPException
from fastapi import UploadFile

_CHUNK_SIZE = 1024 * 1024


async def read_upload_within_limit(file: UploadFile, max_bytes: int) -> bytes:
    """Read `file` into memory, raising a 413 the moment the total exceeds `max_bytes` instead of after buffering the whole upload."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise MessageHTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {max_bytes // (1024 * 1024)}MB",
            )
        chunks.append(chunk)
    return b"".join(chunks)
