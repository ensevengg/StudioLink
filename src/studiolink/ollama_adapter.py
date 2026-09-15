from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from studiolink.config import StudioLinkConfig
from studiolink.models import OllamaModel, OllamaScanReport

MODEL_MEDIA_TYPE = "application/vnd.ollama.image.model"
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

        model_layer = None
        for layer in layers:
            if isinstance(layer, dict) and layer.get("mediaType") == MODEL_MEDIA_TYPE:
                model_layer = layer
                break

        model_digest = None
        blob_path = None
        declared_size = None
        blob_size = None
        gguf_valid = False

        if model_layer is None:
            issues.append("missing Ollama model layer")
        else:
            model_digest = str(model_layer.get("digest", "")).strip() or None
            declared_size = self._as_int(model_layer.get("size"))
            if model_digest is None:
                issues.append("model layer is missing a digest")
            elif SHA256_DIGEST_PATTERN.fullmatch(model_digest) is None:
                issues.append("model layer has an invalid SHA-256 digest")
            else:
                blob_path = self.config.ollama_blobs_dir / model_digest.replace(
                    ":", "-"
                )
                logger.debug("Looking for blob: %s", blob_path)
                if not blob_path.exists():
                    logger.debug("Blob not found: %s", blob_path)
                    issues.append("model blob is missing from the Ollama blob store")
                else:
                    blob_size = blob_path.stat().st_size
                    logger.debug("Found blob: %s (size=%d bytes)", blob_path, blob_size)
                    if self._has_gguf_header(blob_path):
                        logger.debug("Blob has valid GGUF header: %s", blob_path)
                        gguf_valid = True
                    else:
                        logger.debug("Blob missing GGUF magic bytes: %s", blob_path)
                        issues.append("blob does not start with GGUF magic bytes")

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
            model_digest=model_digest,
            blob_path=blob_path,
            declared_size=declared_size,
            blob_size=blob_size,
            gguf_valid=gguf_valid,
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
