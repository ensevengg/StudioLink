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


class ArtifactRole(StrEnum):
    MODEL = "model"
    PROJECTOR = "projector"


@dataclass(slots=True, frozen=True)
class OllamaArtifact:
    role: ArtifactRole
    media_type: str
    digest: str | None
    blob_path: Path | None
    declared_size: int | None
    blob_size: int | None
    gguf_valid: bool
    manifest_index: int
    issues: tuple[str, ...] = ()

    @property
    def readiness(self) -> ModelReadiness:
        if self.gguf_valid:
            return ModelReadiness.READY
        if any("blob is missing" in issue for issue in self.issues):
            return ModelReadiness.STALE
        return ModelReadiness.INVALID


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
    artifacts: tuple[OllamaArtifact, ...] = ()

    @property
    def readiness(self) -> ModelReadiness:
        if self.gguf_valid:
            return ModelReadiness.READY
        if any("not supported" in issue for issue in self.issues):
            return ModelReadiness.INVALID
        if self.artifacts:
            if any(
                artifact.readiness is ModelReadiness.INVALID
                for artifact in self.artifacts
            ):
                return ModelReadiness.INVALID
            if any(
                artifact.readiness is ModelReadiness.STALE
                for artifact in self.artifacts
            ):
                return ModelReadiness.STALE
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
    def import_user_repo(self) -> str:
        if not any(
            artifact.role is ArtifactRole.PROJECTOR for artifact in self.artifacts
        ):
            return self.user_repo
        identity = "-".join(
            (artifact.digest or "unknown").split(":")[-1][:12]
            for artifact in self.artifacts
        )
        return f"{self.user_repo}--bundle-{identity}"

    @property
    def import_filename(self) -> str:
        return self._import_filename(self.model_digest)

    def artifact_import_filename(self, artifact: OllamaArtifact) -> str:
        prefix = "mmproj-" if artifact.role is ArtifactRole.PROJECTOR else ""
        return prefix + self._import_filename(artifact.digest)

    def _import_filename(self, digest: str | None) -> str:
        # Keep the hash portion of the digest so re-pulled models (same name,
        # new digest) get a distinct alias instead of reusing the stale one.
        digest_hash = (digest or "unknown").split(":")[-1][:12]
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
class SyncedArtifact:
    role: ArtifactRole
    digest: str
    blob_path: Path
    import_alias_path: Path
    imported_model_path: Path | None = None
    import_command: tuple[str, ...] = field(default_factory=tuple)

    def to_json(self) -> dict[str, object]:
        return {
            "role": self.role.value,
            "digest": self.digest,
            "blob_path": str(self.blob_path),
            "import_alias_path": str(self.import_alias_path),
            "imported_model_path": (
                None
                if self.imported_model_path is None
                else str(self.imported_model_path)
            ),
            "import_command": list(self.import_command),
        }

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> "SyncedArtifact":
        import_command = payload.get("import_command", [])
        if not isinstance(import_command, list):
            raise TypeError("import_command must be a list")
        return cls(
            role=ArtifactRole(str(payload["role"])),
            digest=str(payload["digest"]),
            blob_path=Path(str(payload["blob_path"])),
            import_alias_path=Path(str(payload["import_alias_path"])),
            imported_model_path=(
                Path(str(payload["imported_model_path"]))
                if payload.get("imported_model_path")
                else None
            ),
            import_command=tuple(str(item) for item in import_command),
        )


@dataclass(slots=True, frozen=True)
class SyncRecord:
    canonical_name: str
    user_repo: str
    link_mode: LinkMode
    imported_at: datetime
    artifacts: tuple[SyncedArtifact, ...]
    vision_confirmed: bool = True

    @property
    def primary_artifact(self) -> SyncedArtifact:
        return next(
            artifact
            for artifact in self.artifacts
            if artifact.role is ArtifactRole.MODEL
        )

    # Compatibility properties for callers and schema-v1 JSON consumers.
    @property
    def digest(self) -> str:
        return self.primary_artifact.digest

    @property
    def blob_path(self) -> Path:
        return self.primary_artifact.blob_path

    @property
    def import_alias_path(self) -> Path:
        return self.primary_artifact.import_alias_path

    @property
    def imported_model_path(self) -> Path | None:
        return self.primary_artifact.imported_model_path

    @property
    def import_command(self) -> tuple[str, ...]:
        return self.primary_artifact.import_command

    def to_json(self) -> dict[str, object]:
        primary = self.primary_artifact
        return {
            "canonical_name": self.canonical_name,
            "user_repo": self.user_repo,
            "link_mode": self.link_mode.value,
            "imported_at": self.imported_at.isoformat(),
            "artifacts": [artifact.to_json() for artifact in self.artifacts],
            "vision_confirmed": self.vision_confirmed,
            # Keep the original primary-artifact fields additive for CLI JSON
            # consumers while state schema v2 adopts the artifacts collection.
            "digest": primary.digest,
            "blob_path": str(primary.blob_path),
            "import_alias_path": str(primary.import_alias_path),
            "imported_model_path": (
                None
                if primary.imported_model_path is None
                else str(primary.imported_model_path)
            ),
            "import_command": list(primary.import_command),
        }

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> "SyncRecord":
        artifacts_payload = payload.get("artifacts")
        if artifacts_payload is not None:
            if not isinstance(artifacts_payload, list):
                raise TypeError("artifacts must be a list")
            artifacts = tuple(
                SyncedArtifact.from_json(item)
                for item in artifacts_payload
                if isinstance(item, dict)
            )
            if len(artifacts) != len(artifacts_payload) or not artifacts:
                raise TypeError("artifacts must contain artifact objects")
        else:
            import_command = payload.get("import_command", [])
            if not isinstance(import_command, list):
                raise TypeError("import_command must be a list")
            artifacts = (
                SyncedArtifact(
                    role=ArtifactRole.MODEL,
                    digest=str(payload["digest"]),
                    blob_path=Path(str(payload["blob_path"])),
                    import_alias_path=Path(str(payload["import_alias_path"])),
                    imported_model_path=(
                        Path(str(payload["imported_model_path"]))
                        if payload.get("imported_model_path")
                        else None
                    ),
                    import_command=tuple(str(item) for item in import_command),
                ),
            )
        if sum(artifact.role is ArtifactRole.MODEL for artifact in artifacts) != 1:
            raise ValueError("sync record must contain exactly one model artifact")
        if sum(artifact.role is ArtifactRole.PROJECTOR for artifact in artifacts) > 1:
            raise ValueError("sync record cannot contain multiple projector artifacts")
        return cls(
            canonical_name=str(payload["canonical_name"]),
            user_repo=str(payload["user_repo"]),
            link_mode=LinkMode(str(payload["link_mode"])),
            imported_at=datetime.fromisoformat(str(payload["imported_at"])),
            artifacts=artifacts,
            vision_confirmed=(
                payload.get("vision_confirmed") is True
                if any(
                    artifact.role is ArtifactRole.PROJECTOR for artifact in artifacts
                )
                else True
            ),
        )


def is_synced(
    model: OllamaModel,
    record: SyncRecord | None,
    *,
    expected_target: Path | None = None,
) -> bool:
    """Check if state and every imported LM Studio artifact match the model."""
    if record is None:
        return False
    expected = (
        tuple((artifact.role, artifact.digest) for artifact in model.artifacts)
        if model.artifacts
        else ((ArtifactRole.MODEL, model.model_digest),)
    )
    actual = tuple((artifact.role, artifact.digest) for artifact in record.artifacts)
    if actual != expected:
        return False
    if any(role is ArtifactRole.PROJECTOR for role, _ in expected):
        if not record.vision_confirmed:
            return False
    for artifact in record.artifacts:
        target = artifact.imported_model_path
        if artifact.role is ArtifactRole.MODEL and target is None:
            target = expected_target
        if target is None:
            if artifact.role is ArtifactRole.PROJECTOR:
                return False
            continue
        if not target.is_file():
            return False
    return True


@dataclass(slots=True, frozen=True)
class LMStudioModel:
    model_key: str
    path: str
    vision: bool
    model_type: str


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
