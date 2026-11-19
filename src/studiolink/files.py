"""Verify and remove only files belonging to a tracked import."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from studiolink.models import LinkMode, OllamaArtifact, SyncedArtifact, SyncRecord

logger = logging.getLogger("studiolink")


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


def matches_identity(path: Path, recorded: tuple[int, int, int, int]) -> bool:
    """Whether ``path`` is still the file whose identity was recorded.

    The device number is stored but never compared: Linux renumbers devices
    across reboots (btrfs subvolumes get anonymous ids at mount time, NVMe
    drives can enumerate in a different order), which would disown every
    import. Inode, size and nanosecond mtime at a fixed path still pin the file.
    """
    return identity(path)[1:] == recorded[1:]


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


def _contained(root: Path, path: Path) -> Path:
    """Spell ``path`` under the configured root, rejecting escapes.

    Only the parent is resolved: the imported file itself may be a symlink to
    Ollama. Re-expressing the resolved parent under ``root`` gives one spelling
    per destination even when the root or a directory below it is a symlink,
    so destinations compare equal wherever they were recorded.
    """
    try:
        relative = path.parent.resolve().relative_to(root.resolve())
    except ValueError:
        raise ValueError(
            f"Import directory resolves outside the configured models directory: {path}"
        ) from None
    return root.absolute().joinpath(relative, path.name)


def destination_path(root: Path, user_repo: str, filename: str) -> Path:
    parts = user_repo.split("/")
    if len(parts) != 2 or any(p in ("", ".", "..") or "\\" in p for p in parts):
        raise ValueError(f"Invalid import repository: {user_repo}")
    return _contained(root, root.absolute().joinpath(*parts, Path(filename).name))


def artifact_destination(
    record: SyncRecord, artifact: SyncedArtifact, root: Path
) -> Path:
    if artifact.imported_model_path is not None:
        try:
            return _contained(root, artifact.imported_model_path)
        except ValueError:
            # LM Studio's library may have moved since the record was saved.
            # Check its expected location under the new root; never delete
            # the old path.
            pass
    return destination_path(root, record.user_repo, artifact.import_alias_path.name)


def owns_destination(record: SyncRecord, artifact: SyncedArtifact, path: Path) -> bool:
    from studiolink.models import LinkMode

    try:
        if artifact.imported_identity is not None:
            return matches_identity(path, artifact.imported_identity)
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
    path: Path,
    blob: Path,
    mode: LinkMode,
    artifact: SyncedArtifact | None = None,
    *,
    compare_contents: bool = True,
) -> bool:
    """Check an import against its blob.

    For copies without a recorded identity this reads both files in full, so
    callers that just watched ``lms`` write the copy pass
    ``compare_contents=False``: the type and size checks are enough there, and
    the identity recorded afterwards makes later checks cheap.
    """
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
            return matches_identity(path, artifact.imported_identity)
        return not compare_contents or equal_contents(path, blob)
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
            and matches_identity(path, artifact.imported_identity)
        )
    except OSError:
        return False


def remove_empty_parents(path: Path, root: Path) -> None:
    """Remove the now-empty repository folders an import lived in, below ``root``."""
    root = root.absolute()
    parent = path.absolute().parent
    while parent != root and root in parent.parents:
        try:
            parent.rmdir()
        except OSError:
            return
        parent = parent.parent


def remove_import(
    record: SyncRecord, root: Path, staging: Path, *, keep: set[Path]
) -> None:
    """Remove an import's files, sparing any a user's own symlink still needs."""
    references = symlink_references(root)
    removed: set[Path] = set()

    def referenced(path: Path) -> bool:
        return any(
            ref.absolute() not in removed and ref.is_symlink()
            for ref in references.get(path.absolute(), ())
        )

    for artifact in record.artifacts:
        path = artifact_destination(record, artifact, root)
        if path not in keep and (path.exists() or path.is_symlink()):
            if not owns_destination(record, artifact, path):
                raise RuntimeError(
                    f"Refusing to remove a changed or unverified import: {path}"
                )
            if referenced(path):
                # A symlink StudioLink did not make still points here; deleting
                # it could destroy the last copy of the weights.
                logger.warning(
                    "Keeping %s: a symbolic link in LM Studio still points to it",
                    path,
                )
                continue
            path.unlink()
            removed.add(path.absolute())
            remove_empty_parents(path, root)
    for artifact in record.artifacts:
        alias = artifact.import_alias_path
        if alias.parent.resolve() == staging.resolve() and alias not in keep:
            if not referenced(alias):
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


def existing_ancestor(path: Path) -> Path:
    """The nearest directory that exists, where a probe stands in for ``path``."""
    path = path.absolute()
    while not path.is_dir() and path != path.parent:
        path = path.parent
    return path


def _probe(directory: Path, create) -> OSError | None:
    probe = existing_ancestor(directory) / f".studiolink-probe-{os.getpid()}"
    try:
        probe.unlink(missing_ok=True)
        create(probe)
    except OSError as exc:
        return exc
    finally:
        try:
            probe.unlink(missing_ok=True)
        except OSError:
            pass
    return None


def probe_hard_link(source: Path, directory: Path) -> OSError | None:
    """Try hard-linking ``source`` into ``directory`` (or the ancestor it will be created in).

    A real link catches what device ids cannot: btrfs subvolumes and bind
    mounts on one disk, and Linux fs.protected_hardlinks on blobs owned by the
    ``ollama`` service user.
    """
    return _probe(directory, lambda probe: os.link(source, probe))


def probe_symlink(source: Path, directory: Path) -> OSError | None:
    """Try creating a symlink to ``source`` (Windows needs Developer Mode)."""
    return _probe(directory, lambda probe: probe.symlink_to(source.absolute()))


def choose_link_mode(
    source: Path | None, directories: list[Path]
) -> tuple[LinkMode, str]:
    """Pick the cheapest import mode that works from ``source`` into ``directories``.

    Hard links cost no space and survive Ollama deleting the blob; symlinks
    cost no space but break if the blob goes; copies always work.
    """
    from studiolink.models import LinkMode

    if source is None:
        return LinkMode.HARD_LINK, "no Ollama blobs to probe yet"
    failures = [
        f"{directory}: {exc.strerror or exc}"
        for directory in directories
        if (exc := probe_hard_link(source, directory)) is not None
    ]
    if not failures:
        return LinkMode.HARD_LINK, "hard links work"
    reason = "hard links unavailable (" + "; ".join(failures) + ")"
    symlink_failure = next(
        (exc for directory in directories if (exc := probe_symlink(source, directory))),
        None,
    )
    if symlink_failure is None:
        return LinkMode.SYMBOLIC_LINK, reason
    return (
        LinkMode.COPY,
        f"{reason}; symlinks unavailable ({symlink_failure.strerror or symlink_failure})",
    )
