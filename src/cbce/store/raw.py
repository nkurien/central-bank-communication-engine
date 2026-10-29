"""Content-addressed store for raw files. Files are written once and never changed."""

import hashlib
import os
import stat
from pathlib import Path


def content_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def raw_path(root: Path, digest: str) -> Path:
    return root / digest[:2] / digest


def put(root: Path, content: bytes) -> tuple[str, Path]:
    """Store `content` under its hash and return (hash, path).

    Idempotent: storing the same bytes again is a no-op. An existing file is never
    rewritten, and one whose bytes no longer match its name raises.
    """
    digest = content_hash(content)
    path = raw_path(root, digest)
    if path.exists():
        if content_hash(path.read_bytes()) != digest:
            raise RuntimeError(f"raw file {path} does not match its hash")
        return digest, path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(content)
    tmp.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    os.replace(tmp, path)
    return digest, path


def get(root: Path, digest: str) -> bytes:
    """Read a stored file, verifying it still matches its hash."""
    content = raw_path(root, digest).read_bytes()
    if content_hash(content) != digest:
        raise RuntimeError(f"raw file for {digest} is corrupt")
    return content
