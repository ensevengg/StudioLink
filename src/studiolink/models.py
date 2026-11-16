from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path


class LinkMode(StrEnum):
    HARD_LINK = "hard-link"
    COPY = "copy"
    SYMBOLIC_LINK = "symbolic-link"

    @property
    def lms_flag(self) -> str:
        return {
            LinkMode.HARD_LINK: "--hard-link",
            LinkMode.COPY: "--copy",
            LinkMode.SYMBOLIC_LINK: "--symbolic-link",
        }[self]


class ImportMode(StrEnum):
    ALIAS = "alias"  # Stage a .gguf hard link, then import that - default
    DIRECT = "direct"  # Import the Ollama blob file directly (no staging alias)


class ModelReadiness(StrEnum):
    READY = "ready"
    STALE = "stale"
    INVALID = "invalid"


@dataclass(slots=True, frozen=True)
class OllamaModel:
    canonical_name: str
    fully_qualified_name: str
    registry: str
    namespace: str
    repository: str
    tag: str
    manifest_path: Path
    model_digest: str | None
    blob_path: Path | None
    declared_size: int | None
    blob_size: int | None
    gguf_valid: bool
    issues: tuple[str, ...] = ()

    @property
    def readiness(self) -> ModelReadiness:
        if self.gguf_valid:
            return ModelReadiness.READY
        if any(
            "blob is missing from the Ollama blob store" in issue
            for issue in self.issues
        ):
            return ModelReadiness.STALE
        return ModelReadiness.INVALID

    @property
    def blob_present(self) -> bool:
        return self.blob_path is not None and self.blob_path.exists()

    @property
    def short_name(self) -> str:
        return f"{self.repository}:{self.tag}"

    @property
    def user_repo(self) -> str:
        repo_parts = []
        if self.registry != "registry.ollama.ai":
            repo_parts.append(self.registry.replace(".", "-"))
        if self.namespace != "library":
            repo_parts.append(self.namespace)
        repo_parts.append(self.repository.replace("/", "--"))
        return f"ollama/{'-'.join(repo_parts)}"

    @property
    def import_filename(self) -> str:
        # Keep the hash portion of the digest so re-pulled models (same name,
        # new digest) get a distinct alias instead of reusing the stale one.
        digest_hash = (self.model_digest or "unknown").split(":")[-1][:12]
        raw = f"{self.canonical_name}-{digest_hash}"
        safe = "".join(
            ch if ch.isalnum() or ch in ("-", "_", ".") else "-" for ch in raw
        )
        while "--" in safe:
            safe = safe.replace("--", "-")
        return f"{safe}.gguf"


@dataclass(slots=True, frozen=True)
class OllamaScanReport:
    models: tuple[OllamaModel, ...] = ()
    source_available: bool = True
    complete: bool = True
    errors: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class SyncRecord:
    canonical_name: str
    digest: str
    blob_path: Path
    import_alias_path: Path
    user_repo: str
    link_mode: LinkMode
    imported_at: datetime
    import_command: tuple[str, ...] = field(default_factory=tuple)
    imported_model_path: Path | None = None
    imported_identity: tuple[int, int, int, int] | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "canonical_name": self.canonical_name,
            "digest": self.digest,
            "blob_path": str(self.blob_path),
            "import_alias_path": str(self.import_alias_path),
            "user_repo": self.user_repo,
            "link_mode": self.link_mode.value,
            "imported_at": self.imported_at.isoformat(),
            "import_command": list(self.import_command),
            "imported_model_path": str(self.imported_model_path)
            if self.imported_model_path
            else None,
            "imported_identity": list(self.imported_identity)
            if self.imported_identity
            else None,
        }

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> "SyncRecord":
        import_command = payload.get("import_command", [])
        if not isinstance(import_command, list):
            raise TypeError("import_command must be a list")
        raw_identity = payload.get("imported_identity")
        imported_identity = None
        if raw_identity is not None:
            if (
                not isinstance(raw_identity, list)
                or len(raw_identity) != 4
                or not all(isinstance(value, int) for value in raw_identity)
            ):
                raise TypeError("imported_identity must contain four integers")
            imported_identity = (
                raw_identity[0],
                raw_identity[1],
                raw_identity[2],
                raw_identity[3],
            )
        return cls(
            canonical_name=str(payload["canonical_name"]),
            digest=str(payload["digest"]),
            blob_path=Path(str(payload["blob_path"])),
            import_alias_path=Path(str(payload["import_alias_path"])),
            user_repo=str(payload["user_repo"]),
            link_mode=LinkMode(str(payload["link_mode"])),
            imported_at=datetime.fromisoformat(str(payload["imported_at"])),
            imported_model_path=Path(str(payload["imported_model_path"]))
            if payload.get("imported_model_path")
            else None,
            imported_identity=imported_identity,
            import_command=tuple(str(item) for item in import_command),
        )


def is_synced(
    model: OllamaModel, record: SyncRecord | None, models_dir: Path | None = None
) -> bool:
    """A saved digest is insufficient: the imported file must still work."""
    from studiolink.files import healthy_destination, record_destination

    if (
        record is None
        or record.digest != model.model_digest
        or model.readiness is not ModelReadiness.READY
        or model.blob_path is None
        or not model.blob_present
    ):
        return False
    try:
        target = (
            record_destination(record, models_dir)
            if models_dir
            else record.imported_model_path
        )
        return target is not None and healthy_destination(
            target, model.blob_path, record.link_mode, record
        )
    except (OSError, ValueError):
        return False


@dataclass(slots=True, frozen=True)
class ImportResult:
    command: tuple[str, ...]
    stdout: str
    stderr: str
    return_code: int
    dry_run: bool = False


@dataclass(slots=True, frozen=True)
class SyncResult:
    model: OllamaModel
    status: str
    message: str
    record: SyncRecord | None = None


@dataclass(slots=True, frozen=True)
class DoctorCheck:
    name: str
    ok: bool
    details: str


@dataclass(slots=True, frozen=True)
class PruneResult:
    """A staging alias eligible for removal (or reported in a dry run)."""

    path: Path
    size: int
    would_free: bool  # alias is the last link, so deleting it frees disk space
    reason: str
    removed: bool

    def to_json(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "size": self.size,
            "would_free": self.would_free,
            "reason": self.reason,
            "removed": self.removed,
        }


@dataclass(slots=True, frozen=True)
class PruneReport:
    dry_run: bool
    aliases: tuple[PruneResult, ...] = ()
    records_removed: tuple[str, ...] = ()

    def to_json(self) -> dict[str, object]:
        return {
            "dry_run": self.dry_run,
            "aliases": [item.to_json() for item in self.aliases],
            "records_removed": list(self.records_removed),
        }
