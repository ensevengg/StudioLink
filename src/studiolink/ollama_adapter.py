from __future__ import annotations

import json
import logging
import re
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
GGUF_MAGIC = b"GGUF"
SHA256_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-fA-F]{64}$")

logger = logging.getLogger("studiolink")


class OllamaAdapter:
    def __init__(self, config: StudioLinkConfig) -> None:
        self.config = config

    def scan_models(self) -> list[OllamaModel]:
        return list(self.scan_report().models)

    def scan_report(self) -> OllamaScanReport:
        manifests_dir = self.config.ollama_manifests_dir
        logger.debug("Scanning manifests directory: %s", manifests_dir)
        if not manifests_dir.is_dir():
            message = f"Manifests directory does not exist: {manifests_dir}"
            logger.warning(message)
            return OllamaScanReport(
                source_available=False,
                complete=False,
                errors=(message,),
            )

        models: list[OllamaModel] = []
        errors: list[str] = []
        try:
            manifest_files = sorted(
                path for path in manifests_dir.rglob("*") if path.is_file()
            )
        except OSError as exc:
            message = f"Could not enumerate Ollama manifests: {exc}"
            logger.warning(message)
            return OllamaScanReport(
                source_available=False,
                complete=False,
                errors=(message,),
            )

        logger.debug("Found %d manifest file(s)", len(manifest_files))
        for manifest_path in manifest_files:
            try:
                model = self._parse_manifest(manifest_path)
            except (OSError, UnicodeError) as exc:
                message = f"Could not inspect manifest {manifest_path}: {exc}"
                logger.warning(message)
                errors.append(message)
                continue
            if model is not None:
                models.append(model)
                logger.debug(
                    "Parsed model: %s (readiness=%s)",
                    model.canonical_name,
                    model.readiness,
                )
            else:
                logger.debug("Skipped manifest: %s", manifest_path)
        return OllamaScanReport(
            models=tuple(models),
            source_available=True,
            complete=not errors,
            errors=tuple(errors),
        )

    def _parse_manifest(self, manifest_path: Path) -> OllamaModel | None:
        logger.debug("Parsing manifest: %s", manifest_path)
        try:
            relative_parts = manifest_path.relative_to(
                self.config.ollama_manifests_dir
            ).parts
        except ValueError:
            logger.debug("Skipping manifest outside manifests dir: %s", manifest_path)
            return None

        if len(relative_parts) < 4:
            logger.debug(
                "Skipping manifest with insufficient path parts: %s", manifest_path
            )
            return None

        registry = relative_parts[0]
        namespace = relative_parts[1]
        repository = "/".join(relative_parts[2:-1])
        tag = relative_parts[-1]
        logger.debug(
            "Manifest parsed: registry=%s, namespace=%s, repository=%s, tag=%s",
            registry,
            namespace,
            repository,
            tag,
        )

        issues: list[str] = []
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.debug("Invalid manifest JSON in %s: %s", manifest_path, exc)
            issues.append(f"invalid manifest JSON: {exc}")
            manifest = {}
        if not isinstance(manifest, dict):
            logger.debug("Manifest is not a JSON object: %s", manifest_path)
            issues.append("manifest is not a JSON object")
            manifest = {}

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
        if model_count == 0:
            issues.append("missing Ollama model layer")
        elif model_count > 1:
            issues.append("multiple Ollama model layers are not supported")
        if projector_count > 1:
            issues.append("multiple Ollama projector layers are not supported")
        for role in sorted(unsupported_roles):
            issues.append(f"Ollama {role} layers are not supported")

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
                bool(artifacts)
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
            if not blob_path.exists():
                issues.append(f"{label} blob is missing from the Ollama blob store")
            else:
                blob_size = blob_path.stat().st_size
                gguf_valid = self._has_gguf_header(blob_path)
                if not gguf_valid:
                    issues.append(
                        "blob does not start with GGUF magic bytes"
                        if role is ArtifactRole.MODEL
                        else f"{label} blob does not start with GGUF magic bytes"
                    )
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
        except (TypeError, ValueError):
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
        if registry == "registry.ollama.ai":
            return base_name
        return f"{registry}/{base_name}"
