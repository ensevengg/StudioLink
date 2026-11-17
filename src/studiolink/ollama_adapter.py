from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

from studiolink.config import StudioLinkConfig
from studiolink.models import (
    ArtifactRole,
    OllamaArtifact,
    OllamaModel,
    OllamaScanReport,
)

MODEL_MEDIA_TYPE = "application/vnd.ollama.image.model"
PROJECTOR_MEDIA_TYPE = "application/vnd.ollama.image.projector"
UNSUPPORTED_MEDIA_TYPES = {
    "application/vnd.ollama.image.adapter": "adapter",
    "application/vnd.ollama.image.draft": "draft",
    "application/vnd.ollama.image.tensor": "tensor",
}
MANIFEST_MEDIA_TYPE = "application/vnd.docker.distribution.manifest.v2+json"
MANIFEST_LIST_MEDIA_TYPE = "application/vnd.ollama.manifest.list.v2+json"
# Ollama 0.40+ manifest lists hold one child per runner. llamacpp children are
# the clean llama.cpp-compatible GGUF that LM Studio can load; ggml children are
# Ollama's original GGUF layout, which older StudioLink releases imported.
GGUF_RUNNER_PREFERENCE = ("llamacpp", "ggml")
MLX_RUNNER = "mlx"
SAFETENSORS_FORMAT = "safetensors"
MLX_ONLY_ISSUE = (
    "Ollama MLX (safetensors) models are not supported; LM Studio imports need "
    "a GGUF build"
)
GGUF_MAGIC = b"GGUF"
SHA256_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-fA-F]{64}$")
BLOB_FILENAME_PATTERN = re.compile(r"^sha256-([0-9a-fA-F]{64})$")
HEX_DIGEST_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
DEFAULT_REGISTRY = "registry.ollama.ai"
# manifests-v2 stores the default public registry under its canonical host.
PUBLIC_REGISTRY_HOSTS = frozenset({DEFAULT_REGISTRY, "ollama.com"})
LEGACY_LAYOUT = "legacy"
V2_LAYOUT = "v2"


@dataclass(slots=True, frozen=True)
class _ManifestEntry:
    registry: str
    namespace: str
    repository: str
    tag: str
    path: Path
    layout: str

    @property
    def key(self) -> tuple[str, str, str, str]:
        # Ollama compares names case-insensitively.
        return (
            self.registry.lower(),
            self.namespace.lower(),
            self.repository.lower(),
            self.tag.lower(),
        )

    @property
    def is_converted_shadow(self) -> bool:
        # Ollama's compat migration writes a legacy "llamacpp:<digest>" tag
        # per converted child as a downgrade anchor; it is not a user model.
        return (
            self.registry == DEFAULT_REGISTRY
            and self.namespace == "library"
            and self.repository == "llamacpp"
            and HEX_DIGEST_PATTERN.fullmatch(self.tag) is not None
        )


class _ManifestBlobMissing(Exception):
    pass


logger = logging.getLogger("studiolink")


class OllamaAdapter:
    def __init__(self, config: StudioLinkConfig) -> None:
        self.config = config

    def scan_models(self) -> list[OllamaModel]:
        return list(self.scan_report().models)

    def scan_report(self) -> OllamaScanReport:
        roots = (
            (V2_LAYOUT, self.config.ollama_manifests_v2_path),
            (LEGACY_LAYOUT, self.config.ollama_manifests_dir),
        )
        present = [(layout, root) for layout, root in roots if root.is_dir()]
        if not present:
            message = "Manifests directory does not exist: " + " or ".join(
                str(root) for _, root in roots
            )
            logger.warning(message)
            return OllamaScanReport(
                source_available=False,
                complete=False,
                errors=(message,),
            )

        errors: list[str] = []
        entries: dict[str, dict[tuple[str, str, str, str], _ManifestEntry]] = {}
        enumerated = 0
        for layout, root in present:
            logger.debug("Scanning %s manifests directory: %s", layout, root)
            try:
                paths = self._enumerate(root)
            except OSError as exc:
                message = f"Could not enumerate Ollama manifests in {root}: {exc}"
                logger.warning(message)
                errors.append(message)
                continue
            enumerated += 1
            logger.debug("Found %d %s manifest file(s)", len(paths), layout)
            found = entries.setdefault(layout, {})
            for path in paths:
                entry = self._entry_for_path(root, path, layout)
                if entry is None:
                    logger.debug("Skipped manifest: %s", path)
                elif entry.is_converted_shadow:
                    logger.debug("Skipped converted shadow manifest: %s", path)
                else:
                    found.setdefault(entry.key, entry)
        if not enumerated:
            return OllamaScanReport(
                source_available=False,
                complete=False,
                errors=tuple(errors),
            )

        v2_entries = entries.get(V2_LAYOUT, {})
        legacy_entries = entries.get(LEGACY_LAYOUT, {})
        selected: list[_ManifestEntry] = []
        for key in sorted(v2_entries.keys() | legacy_entries.keys()):
            v2 = v2_entries.get(key)
            legacy = legacy_entries.get(key)
            # Like Ollama, prefer manifests-v2 and fall back to the legacy
            # tree only when the v2 name is absent or a dangling link.
            if v2 is not None and (legacy is None or v2.path.exists()):
                selected.append(v2)
            elif legacy is not None:
                selected.append(legacy)

        models: list[OllamaModel] = []
        for entry in selected:
            try:
                model = self._parse_manifest(entry)
            except (OSError, UnicodeError) as exc:
                message = f"Could not inspect manifest {entry.path}: {exc}"
                logger.warning(message)
                errors.append(message)
                continue
            models.append(model)
            logger.debug(
                "Parsed model: %s (readiness=%s, layout=%s)",
                model.canonical_name,
                model.readiness,
                entry.layout,
            )
        return OllamaScanReport(
            models=tuple(models),
            source_available=True,
            complete=not errors,
            errors=tuple(errors),
        )

    @staticmethod
    def _enumerate(root: Path) -> list[Path]:
        def scan_error(exc: OSError) -> None:
            raise exc

        # os.walk lists symlinks (v2 names are links to manifest blobs) among
        # the file names without following directory links.
        return sorted(
            Path(directory) / name
            for directory, _, names in os.walk(root, onerror=scan_error)
            for name in names
        )

    @staticmethod
    def _entry_for_path(root: Path, path: Path, layout: str) -> _ManifestEntry | None:
        try:
            parts = path.relative_to(root).parts
        except ValueError:
            return None
        if layout == V2_LAYOUT:
            # manifests-v2/<host>/<namespace>/<model>/<tag>, with ":" in the
            # host percent-encoded so ports survive on every filesystem.
            if len(parts) != 4:
                return None
            registry = re.sub("%3[aA]", ":", parts[0])
            repository = parts[2]
        else:
            if len(parts) < 4:
                return None
            registry = parts[0]
            repository = "/".join(parts[2:-1])
        if registry.lower() in PUBLIC_REGISTRY_HOSTS:
            # One identity for the public registry keeps canonical names,
            # LM Studio repos, and sync records stable across layouts.
            registry = DEFAULT_REGISTRY
        return _ManifestEntry(
            registry=registry,
            namespace=parts[1],
            repository=repository,
            tag=parts[-1],
            path=path,
            layout=layout,
        )

    def _read_manifest(self, path: Path) -> object:
        """Return the parsed manifest document at ``path``.

        Raises ``_ManifestBlobMissing`` for a dangling v2 link and
        ``ValueError`` (via ``json``) for malformed content.
        """
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            if path.is_symlink():
                raise _ManifestBlobMissing from None
            raise
        if path.is_symlink():
            match = BLOB_FILENAME_PATTERN.fullmatch(path.resolve().name)
            if match is None or hashlib.sha256(data).hexdigest() != (
                match.group(1).lower()
            ):
                raise ValueError("manifest link does not match its content digest")
        return json.loads(data.decode("utf-8"))

    def _read_manifest_blob(self, digest: str) -> object:
        blob_path = self.config.ollama_blobs_dir / digest.replace(":", "-")
        try:
            data = blob_path.read_bytes()
        except FileNotFoundError:
            raise _ManifestBlobMissing from None
        if "sha256:" + hashlib.sha256(data).hexdigest() != digest.lower():
            raise ValueError("child manifest blob does not match its digest")
        return json.loads(data.decode("utf-8"))

    def _select_child(self, manifest: dict[str, object]) -> tuple[object, list[str]]:
        """Pick the GGUF child of an Ollama manifest list for LM Studio.

        Returns the child document, or ``None`` with the issues explaining why
        no importable child exists.
        """
        children = manifest.get("manifests")
        if not isinstance(children, list):
            return None, ["manifest list entries must be a list"]
        candidates = [
            child
            for child in children
            if isinstance(child, dict)
            and child.get("mediaType") in (None, "", MANIFEST_MEDIA_TYPE)
        ]
        missing = False
        problems: list[str] = []
        for runner in GGUF_RUNNER_PREFERENCE:
            for child in candidates:
                if str(child.get("runner", "")).strip().lower() != runner:
                    continue
                if str(child.get("format", "gguf")).strip().lower() != "gguf":
                    continue
                try:
                    resolved = self._resolve_child(child)
                except _ManifestBlobMissing:
                    missing = True
                    continue
                except (ValueError, UnicodeError) as exc:
                    problems.append(f"invalid {runner} child manifest: {exc}")
                    continue
                if isinstance(resolved, dict) and (
                    resolved.get("mediaType") == MANIFEST_LIST_MEDIA_TYPE
                ):
                    problems.append("nested Ollama manifest lists are not supported")
                    continue
                return resolved, []
        if problems:
            return None, problems
        if missing:
            return None, [
                "GGUF child manifest blob is missing from the Ollama blob store"
            ]
        if any(self._is_mlx(child) for child in candidates):
            return None, [MLX_ONLY_ISSUE]
        return None, ["Ollama manifest lists without a GGUF child are not supported"]

    def _resolve_child(self, child: dict[str, object]) -> object:
        layers = child.get("layers")
        digest = child.get("digest")
        if layers is None and isinstance(digest, str) and digest:
            if SHA256_DIGEST_PATTERN.fullmatch(digest) is None:
                raise ValueError("child manifest has an invalid SHA-256 digest")
            return self._read_manifest_blob(digest)
        return child

    @staticmethod
    def _is_mlx(manifest: dict[str, object]) -> bool:
        return (
            str(manifest.get("runner", "")).strip().lower() == MLX_RUNNER
            or str(manifest.get("format", "")).strip().lower() == SAFETENSORS_FORMAT
        )

    def _parse_manifest(self, entry: _ManifestEntry) -> OllamaModel:
        manifest_path = entry.path
        logger.debug("Parsing manifest: %s", manifest_path)
        registry = entry.registry
        namespace = entry.namespace
        repository = entry.repository
        tag = entry.tag
        logger.debug(
            "Manifest parsed: registry=%s, namespace=%s, repository=%s, tag=%s",
            registry,
            namespace,
            repository,
            tag,
        )

        issues: list[str] = []
        # Selection issues explain the whole model; skip layer diagnostics.
        layer_issues = True
        try:
            manifest = self._read_manifest(manifest_path)
        except _ManifestBlobMissing:
            issues.append("manifest blob is missing from the Ollama blob store")
            manifest = {}
            layer_issues = False
        except UnicodeError:
            # Undecodable bytes are a scan error, not a model verdict.
            raise
        except ValueError as exc:
            # json.JSONDecodeError is a ValueError, as is a digest mismatch.
            logger.debug("Invalid manifest in %s: %s", manifest_path, exc)
            issues.append(
                f"invalid manifest JSON: {exc}"
                if isinstance(exc, json.JSONDecodeError)
                else str(exc)
            )
            manifest = {}
        if not isinstance(manifest, dict):
            logger.debug("Manifest is not a JSON object: %s", manifest_path)
            issues.append("manifest is not a JSON object")
            manifest = {}
        if manifest.get("mediaType") == MANIFEST_LIST_MEDIA_TYPE:
            child, selection_issues = self._select_child(manifest)
            issues.extend(selection_issues)
            if isinstance(child, dict):
                manifest = child
            elif child is None:
                manifest = {}
                layer_issues = False
            else:
                issues.append("child manifest is not a JSON object")
                manifest = {}
                layer_issues = False
        elif self._is_mlx(manifest):
            issues.append(MLX_ONLY_ISSUE)
            layer_issues = False

        layers = manifest.get("layers", [])
        if not isinstance(layers, list):
            issues.append("manifest layers must be a list")
            layers = []

        artifact_layers: list[tuple[int, dict[str, object], ArtifactRole]] = []
        unsupported_roles: set[str] = set()
        for index, layer in enumerate(layers):
            if not isinstance(layer, dict):
                continue
            media_type = layer.get("mediaType")
            if media_type == MODEL_MEDIA_TYPE:
                artifact_layers.append((index, layer, ArtifactRole.MODEL))
            elif media_type == PROJECTOR_MEDIA_TYPE:
                artifact_layers.append((index, layer, ArtifactRole.PROJECTOR))
            elif isinstance(media_type, str) and media_type in UNSUPPORTED_MEDIA_TYPES:
                unsupported_roles.add(UNSUPPORTED_MEDIA_TYPES[media_type])

        model_count = sum(role is ArtifactRole.MODEL for _, _, role in artifact_layers)
        projector_count = sum(
            role is ArtifactRole.PROJECTOR for _, _, role in artifact_layers
        )
        if layer_issues:
            if model_count == 0:
                issues.append("missing Ollama model layer")
            elif model_count > 1:
                issues.append("multiple Ollama model layers are not supported")
            if projector_count > 1:
                issues.append("multiple Ollama projector layers are not supported")
            for role in sorted(unsupported_roles):
                issues.append(f"Ollama {role} layers are not supported")

        # Import order is semantic: partial sync records must always contain
        # the model artifact, so normalize model layers before projectors
        # regardless of the order the manifest lists them in.
        artifact_layers.sort(key=lambda entry: entry[2] is not ArtifactRole.MODEL)
        artifacts = tuple(
            self._parse_artifact(layer, role, index)
            for index, layer, role in artifact_layers
        )
        for artifact in artifacts:
            issues.extend(artifact.issues)
        primary = next(
            (artifact for artifact in artifacts if artifact.role is ArtifactRole.MODEL),
            None,
        )

        canonical_name = self._canonical_name(registry, namespace, repository, tag)
        fully_qualified_name = f"{registry}/{namespace}/{repository}:{tag}"
        return OllamaModel(
            canonical_name=canonical_name,
            fully_qualified_name=fully_qualified_name,
            registry=registry,
            namespace=namespace,
            repository=repository,
            tag=tag,
            manifest_path=manifest_path,
            model_digest=primary.digest if primary else None,
            blob_path=primary.blob_path if primary else None,
            declared_size=primary.declared_size if primary else None,
            blob_size=primary.blob_size if primary else None,
            gguf_valid=(
                layer_issues
                and bool(artifacts)
                and model_count == 1
                and projector_count <= 1
                and not unsupported_roles
                and all(artifact.gguf_valid for artifact in artifacts)
            ),
            issues=tuple(issues),
            artifacts=artifacts,
        )

    def _parse_artifact(
        self, layer: dict[str, object], role: ArtifactRole, manifest_index: int
    ) -> OllamaArtifact:
        label = role.value
        digest = str(layer.get("digest", "")).strip() or None
        declared_size = self._as_int(layer.get("size"))
        blob_path = None
        blob_size = None
        gguf_valid = False
        issues: list[str] = []
        if digest is None:
            issues.append(f"{label} layer is missing a digest")
        elif SHA256_DIGEST_PATTERN.fullmatch(digest) is None:
            issues.append(f"{label} layer has an invalid SHA-256 digest")
        else:
            blob_path = self.config.ollama_blobs_dir / digest.replace(":", "-")
            logger.debug("Looking for %s blob: %s", label, blob_path)
            try:
                blob_size = blob_path.stat().st_size
                gguf_valid = self._has_gguf_header(blob_path)
                if not gguf_valid:
                    issues.append(
                        "blob does not start with GGUF magic bytes"
                        if role is ArtifactRole.MODEL
                        else f"{label} blob does not start with GGUF magic bytes"
                    )
                if declared_size is not None and blob_size != declared_size:
                    issues.append(
                        f"{label} blob size mismatch: "
                        f"expected {declared_size}, found {blob_size}"
                    )
                    gguf_valid = False
            except FileNotFoundError:
                issues.append(f"{label} blob is missing from the Ollama blob store")
            except OSError as exc:
                issues.append(f"could not read {label} blob: {exc}")
        return OllamaArtifact(
            role=role,
            media_type=str(layer.get("mediaType", "")),
            digest=digest,
            blob_path=blob_path,
            declared_size=declared_size,
            blob_size=blob_size,
            gguf_valid=gguf_valid,
            manifest_index=manifest_index,
            issues=tuple(issues),
        )

    @staticmethod
    def _as_int(value: object) -> int | None:
        if not isinstance(value, (str, int, float)):
            return None
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def _has_gguf_header(blob_path: Path) -> bool:
        with blob_path.open("rb") as handle:
            return handle.read(4) == GGUF_MAGIC

    @staticmethod
    def _canonical_name(
        registry: str, namespace: str, repository: str, tag: str
    ) -> str:
        base_name = (
            f"{repository}:{tag}"
            if namespace == "library"
            else f"{namespace}/{repository}:{tag}"
        )
        if registry == DEFAULT_REGISTRY:
            return base_name
        return f"{registry}/{base_name}"
