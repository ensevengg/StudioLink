"""Verify and remove only files belonging to a tracked import."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from studiolink.models import LinkMode, SyncRecord


def same_file(left: Path, right: Path) -> bool:
    try:
        return left.samefile(right)
    except OSError:
        return False


def identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.lstat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def equal_contents(left: Path, right: Path) -> bool:
    """Compare without allocating model-sized buffers or writing more copies."""
    try:
        if left.stat().st_size != right.stat().st_size:
            return False
        with left.open("rb") as a, right.open("rb") as b:
            while chunk := a.read(1024 * 1024):
                if chunk != b.read(len(chunk)):
                    return False
            return not b.read(1)
    except OSError:
        return False


def destination_path(root: Path, user_repo: str, filename: str) -> Path:
    parts = user_repo.split("/")
    if len(parts) != 2 or any(p in ("", ".", "..") or "\\" in p for p in parts):
        raise ValueError(f"Invalid import repository: {user_repo}")
    return root.absolute().joinpath(*parts, Path(filename).name)


def healthy_destination(
    path: Path, blob: Path, mode: LinkMode, record: SyncRecord | None = None
) -> bool:
    from studiolink.models import LinkMode

    try:
        if not path.is_file():
            return False
        if mode is LinkMode.HARD_LINK:
            return not path.is_symlink() and same_file(path, blob)
        if mode is LinkMode.SYMBOLIC_LINK:
            return path.is_symlink() and same_file(path, blob)
        if path.is_symlink() or same_file(path, blob):
            return False
        if path.stat().st_size != blob.stat().st_size:
            return False
        if record is not None and record.imported_identity is not None:
            return identity(path) == record.imported_identity
        return equal_contents(path, blob)
    except OSError:
        return False
