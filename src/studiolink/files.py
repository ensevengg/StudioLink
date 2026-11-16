"""Verify and remove only files belonging to a tracked import."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from studiolink.models import LinkMode, OllamaArtifact, SyncedArtifact, SyncRecord


def link_target(path: Path) -> Path:
    """Return the absolute path a symlink points to, without following chains.

    Windows reports absolute targets in their ``\\\\?\\`` extended-length
    form, which never compares equal to the paths StudioLink records, so strip
    it before resolving the target against the link's directory.
    """
    raw = os.readlink(path)
    if os.name == "nt":
        if raw.startswith("\\\\?\\UNC\\"):
            raw = "\\\\" + raw[len("\\\\?\\UNC\\") :]
        elif raw.startswith("\\\\?\\"):
            raw = raw[len("\\\\?\\") :]
    return Path(os.path.abspath(path.parent / raw))


def same_file(left: Path, right: Path) -> bool:
    """Check whether two paths reference the same inode (hard link)."""
    try:
        left_stat, right_stat = left.stat(), right.stat()
    except OSError:
        return False
    if left_stat.st_ino == 0 or right_stat.st_ino == 0:
        # Filesystems without file ids (FAT/exFAT) report inode 0, which
        # would make every file on the volume compare equal and reuse a
        # stale alias. Identity is unknowable there, so never match.
        return False
    return (left_stat.st_dev, left_stat.st_ino) == (
        right_stat.st_dev,
        right_stat.st_ino,
    )


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


def artifact_destination(
    record: SyncRecord, artifact: SyncedArtifact, root: Path
) -> Path:
    path = artifact.imported_model_path or destination_path(
        root, record.user_repo, artifact.import_alias_path.name
    )
    # Resolve the parent only: the imported file itself may be a symlink to Ollama.
    if not path.parent.resolve().is_relative_to(root.resolve()):
        # LM Studio's library may have moved since the record was saved. Check
        # its expected location under the new root; never delete the old path.
        path = destination_path(root, record.user_repo, artifact.import_alias_path.name)
    if not path.parent.resolve().is_relative_to(root.resolve()):
        raise ValueError(
            f"Import directory resolves outside the configured models directory: {path}"
        )
    return path


def owns_destination(record: SyncRecord, artifact: SyncedArtifact, path: Path) -> bool:
    from studiolink.models import LinkMode

    try:
        if artifact.imported_identity is not None:
            return identity(path) == artifact.imported_identity
        # Legacy records have no file identity. Require evidence before removal.
        if path.is_symlink():
            return link_target(path) == artifact.import_alias_path.absolute()
        for source in (artifact.import_alias_path, artifact.blob_path):
            if same_file(path, source):
                return True
            if record.link_mode is LinkMode.COPY and equal_contents(path, source):
                return True
    except OSError:
        pass
    return False


def healthy_destination(
    path: Path, blob: Path, mode: LinkMode, artifact: SyncedArtifact | None = None
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
        if artifact is not None and artifact.imported_identity is not None:
            return identity(path) == artifact.imported_identity
        return equal_contents(path, blob)
    except OSError:
        return False


def intact_destination(
    path: Path, source: OllamaArtifact, mode: LinkMode, artifact: SyncedArtifact
) -> bool:
    """Check an import against its Ollama blob, or its own identity if that's gone."""
    from studiolink.models import ModelReadiness

    blob = source.blob_path
    if source.readiness is ModelReadiness.INVALID or blob is None:
        return False
    # Check the blob now rather than trusting the scan: it may have gone since.
    if blob.exists():
        return healthy_destination(path, blob, mode, artifact)
    # A hard link or copy keeps working without the blob, but only an unchanged
    # file StudioLink recorded is evidence that it still holds those weights. A
    # symlink's own identity says nothing about the weights it reaches.
    try:
        return (
            artifact.imported_identity is not None
            and path.is_file()
            and not path.is_symlink()
            and identity(path) == artifact.imported_identity
        )
    except OSError:
        return False


def remove_import(
    record: SyncRecord, root: Path, staging: Path, *, keep: set[Path]
) -> None:
    for artifact in record.artifacts:
        path = artifact_destination(record, artifact, root)
        if path not in keep and (path.exists() or path.is_symlink()):
            if not owns_destination(record, artifact, path):
                raise RuntimeError(
                    f"Refusing to remove a changed or unverified import: {path}"
                )
            path.unlink()
    references = symlink_references(root)
    for artifact in record.artifacts:
        alias = artifact.import_alias_path
        if alias.parent.resolve() == staging.resolve() and alias not in keep:
            if alias.absolute() not in references:
                alias.unlink(missing_ok=True)


def symlink_references(root: Path) -> dict[Path, set[Path]]:
    """Find LM Studio symlinks before deleting aliases, including untracked ones."""
    references: dict[Path, set[Path]] = {}

    def fail(exc: OSError) -> None:
        raise exc

    for directory, dirs, files in os.walk(root, onerror=fail):
        for name in (*dirs, *files):
            path = Path(directory) / name
            if path.is_symlink():
                references.setdefault(link_target(path), set()).add(path)
    return references
