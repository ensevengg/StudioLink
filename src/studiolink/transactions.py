"""Recoverable temporary destinations for imports and atomic replacements."""

from __future__ import annotations

import ctypes
import errno
import json
import os
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from studiolink.models import PruneResult
from studiolink.state import StateStore

TRANSACTIONS = ".studiolink-transactions"
MARKER = "owner.json"


def _root(models: Path) -> Path:
    root = models / TRANSACTIONS
    if root.is_symlink():
        raise RuntimeError(f"Transaction directory must not be a symlink: {root}")
    return root


def _cleanup(work: Path, filename: str) -> None:
    (work / filename).unlink(missing_ok=True)
    (work / MARKER).unlink(missing_ok=True)


def _remove_empty(work: Path) -> None:
    (work / "lock.json.lock").unlink(missing_ok=True)
    work.rmdir()


def recover_imports(models: Path, *, dry_run: bool = False) -> tuple[PruneResult, ...]:
    root = _root(models)
    if not root.exists():
        return ()
    results: list[PruneResult] = []
    for work in root.iterdir():
        if (
            work.is_symlink()
            or not work.is_dir()
            or not work.name.startswith("import-")
        ):
            continue
        marker = work / MARKER
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            continue
        filename = payload.get("filename") if isinstance(payload, dict) else None
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or filename in ("", ".", "..", MARKER, "lock.json.lock")
        ):
            continue
        try:
            with StateStore(work / "lock.json").lock():
                target = work / filename
                try:
                    stat = target.lstat()
                except FileNotFoundError:
                    stat = None
                is_symlink = target.is_symlink()
                if not dry_run:
                    _cleanup(work, filename)
                if stat is not None:
                    results.append(
                        PruneResult(
                            target,
                            stat.st_size,
                            not is_symlink and stat.st_nlink == 1,
                            "abandoned import transaction",
                            not dry_run,
                        )
                    )
        except RuntimeError:
            # Another process may still be importing into this directory.
            continue
        if not dry_run:
            _remove_empty(work)
    return tuple(results)


@contextmanager
def import_transaction(models: Path, filename: str) -> Iterator[Path]:
    root = _root(models)
    root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="import-", dir=root))
    try:
        with StateStore(work / "lock.json").lock():
            (work / MARKER).write_text(
                json.dumps({"filename": filename}), encoding="utf-8"
            )
            try:
                yield work / filename
            finally:
                _cleanup(work, filename)
    finally:
        # Never recursively remove a directory containing unknown files.
        if not (work / MARKER).exists():
            _remove_empty(work)


def publish_import(staged: Path, target: Path) -> None:
    """Publish without overwriting a file created by another importer."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if staged.is_symlink():
        target.symlink_to(os.readlink(staged))
    else:
        # Both paths are on the LM Studio filesystem; no second weight copy.
        try:
            os.link(staged, target)
        except OSError as exc:
            if exc.errno not in (errno.EPERM, errno.ENOTSUP):
                raise
            _rename_without_overwrite(staged, target)
    staged.unlink(missing_ok=True)


def _rename_without_overwrite(source: Path, target: Path) -> None:
    # exFAT and some network filesystems support rename but not hard links.
    if os.name == "nt":
        os.rename(source, target)  # Windows rename already refuses replacement.
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        result = libc.renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1)
    elif sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        result = libc.renamex_np(os.fsencode(source), os.fsencode(target), 4)
    else:
        raise OSError(
            errno.ENOTSUP,
            "Filesystem cannot publish an import without overwriting existing files",
        )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(target))
