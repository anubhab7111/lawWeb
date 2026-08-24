"""
Shared streaming-read guard for file uploads. Reading the whole body via
UploadFile.read() and checking its size afterward still materializes an
attacker- or user-controlled amount of data into one Python `bytes` object
first — on a machine with ~15GB RAM shared with Ollama and the
embedding/reranker models, a single large POST can OOM the (single) worker
before that check ever runs. Reading in bounded chunks and aborting the
moment the limit is crossed keeps memory use bounded to max_bytes (+ one
chunk) regardless of how large the upload actually is.
"""

from fastapi import UploadFile

from app.deps.errors import MessageHTTPException

_CHUNK_SIZE = 1024 * 1024  # 1MB


async def read_upload_within_limit(file: UploadFile, max_bytes: int) -> bytes:
    """Read `file` into memory, raising a 413 the moment the total exceeds
    `max_bytes` instead of after buffering the whole upload."""
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
