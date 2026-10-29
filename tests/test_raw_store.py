from pathlib import Path

import pytest

from cbce.store import raw


@pytest.mark.parametrize(
    "content",
    [b"statement text", b"", b"\x00\xff binary", "café".encode()],
    ids=["text", "empty", "binary", "utf8"],
)
def test_put_then_get_round_trips(tmp_path: Path, content: bytes) -> None:
    # Arrange: an empty store

    # Act
    digest, path = raw.put(tmp_path, content)

    # Assert
    assert path == raw.raw_path(tmp_path, digest)
    assert raw.get(tmp_path, digest) == content


@pytest.mark.parametrize("repeats", [2, 5])
def test_put_is_idempotent_and_does_not_rewrite(tmp_path: Path, repeats: int) -> None:
    # Arrange
    first = raw.put(tmp_path, b"same")
    mtime = first[1].stat().st_mtime_ns

    # Act
    results = [raw.put(tmp_path, b"same") for _ in range(repeats)]

    # Assert
    assert all(result == first for result in results)
    assert first[1].stat().st_mtime_ns == mtime


def test_stored_file_is_read_only(tmp_path: Path) -> None:
    # Arrange / Act
    _, path = raw.put(tmp_path, b"locked")

    # Assert
    assert not path.stat().st_mode & 0o222


@pytest.mark.parametrize(
    ("first", "second"),
    [(b"a", b"b"), (b"a", b"a "), (b"", b"\x00")],
)
def test_different_content_gets_different_path(tmp_path: Path, first: bytes, second: bytes) -> None:
    # Arrange / Act
    path_first = raw.put(tmp_path, first)[1]
    path_second = raw.put(tmp_path, second)[1]

    # Assert
    assert path_first != path_second


@pytest.mark.parametrize("operation", ["get", "put"])
def test_tampered_file_is_detected(tmp_path: Path, operation: str) -> None:
    # Arrange
    digest, path = raw.put(tmp_path, b"original")
    path.chmod(0o644)
    path.write_bytes(b"edited")

    # Act / Assert
    with pytest.raises(RuntimeError):
        if operation == "get":
            raw.get(tmp_path, digest)
        else:
            raw.put(tmp_path, b"original")
