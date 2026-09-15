from __future__ import annotations

import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from studiolink.config import StudioLinkConfig
from studiolink.lmstudio_adapter import LMStudioError
from studiolink.models import (
    ArtifactRole,
    ImportMode,
    ImportResult,
    LinkMode,
    ModelReadiness,
    OllamaArtifact,
    OllamaModel,
    SyncedArtifact,
    SyncRecord,
    SyncResult,
    is_synced,
)
from studiolink.ports import LMStudioPort
from studiolink.state import StateStore

logger = logging.getLogger("studiolink")

INVENTORY_ATTEMPTS = 10
INVENTORY_RETRY_DELAY = 0.5


def same_file(left: Path, right: Path) -> bool:
    """Check whether two paths reference the same inode (hard link)."""
    try:
        left_stat, right_stat = left.stat(), right.stat()
    except OSError:
        return False
    return (left_stat.st_dev, left_stat.st_ino) == (
        right_stat.st_dev,
        right_stat.st_ino,
    )


class Syncer:
    def __init__(
        self,
        config: StudioLinkConfig,
        lmstudio: LMStudioPort,
        state: StateStore,
    ) -> None:
        self.config = config
        self.lmstudio = lmstudio
        self.state = state

    def sync(
        self,
        models: list[OllamaModel],
        link_mode: LinkMode,
        import_mode: ImportMode,
        dry_run: bool,
    ) -> list[SyncResult]:
        logger.debug(
            "Syncing %d model(s): link_mode=%s, import_mode=%s, dry_run=%s",
            len(models),
            link_mode,
            import_mode,
            dry_run,
        )
        records = self.state.get_all_records()
        logger.debug("Loaded %d existing sync record(s)", len(records))
        results: list[SyncResult] = []
        for model in models:
            result = self._sync_one(model, records, link_mode, import_mode, dry_run)
            results.append(result)
            if result.record and not dry_run:
                records[result.record.canonical_name] = result.record
                self.state.save(records)
                logger.debug("Saved sync record for %s", model.canonical_name)
        return results

    def _sync_one(
        self,
        model: OllamaModel,
        records: dict[str, SyncRecord],
        link_mode: LinkMode,
        import_mode: ImportMode,
        dry_run: bool,
    ) -> SyncResult:
        logger.debug(
            "Processing model: %s (readiness=%s)",
            model.canonical_name,
            model.readiness,
        )
        existing = records.get(model.canonical_name)
        expected_target = None
        if existing is not None and existing.imported_model_path is None:
            expected_target = (
                self.config.lmstudio_models_dir
                / existing.user_repo
                / existing.import_alias_path.name
            )
        already_synced = is_synced(model, existing, expected_target=expected_target)
        has_projector = any(
            artifact.role is ArtifactRole.PROJECTOR for artifact in model.artifacts
        )
        if already_synced and not has_projector:
            logger.debug("Model %s is already synced, skipping", model.canonical_name)
            return SyncResult(
                model=model,
                status="skipped",
                message="already synced according to StudioLink state",
                record=existing,
            )
        if model.readiness is ModelReadiness.STALE:
            logger.debug("Model %s has a missing artifact", model.canonical_name)
            return SyncResult(
                model=model,
                status="error",
                message=(
                    "model artifact is missing from the Ollama blob store; "
                    f"run `ollama pull {model.canonical_name}` to restore it"
                ),
            )
        if model.readiness is ModelReadiness.INVALID:
            logger.debug("Model %s is invalid: %s", model.canonical_name, model.issues)
            return SyncResult(
                model=model,
                status="error",
                message="model is not ready for import: "
                + "; ".join(model.issues or ("unknown error",)),
            )

        artifacts = self._model_artifacts(model)
        if import_mode is ImportMode.DIRECT and len(artifacts) > 1:
            return SyncResult(
                model=model,
                status="error",
                message=(
                    "--direct is not supported for multi-artifact models; "
                    "projector imports require an mmproj- alias"
                ),
            )

        existing_by_identity = {
            (artifact.role, artifact.digest): artifact
            for artifact in (existing.artifacts if existing is not None else ())
        }
        completed: list[SyncedArtifact] = []
        last_result: ImportResult | None = None
        imported_at = (
            existing.imported_at
            if existing is not None
            else datetime.now(tz=timezone.utc)
        )

        for artifact in artifacts:
            if artifact.digest is None or artifact.blob_path is None:
                return SyncResult(
                    model=model,
                    status="error",
                    message=f"{artifact.role.value} artifact is not importable",
                )
            prior = existing_by_identity.get((artifact.role, artifact.digest))
            prior_target = prior.imported_model_path if prior is not None else None
            if prior is not None and prior_target is None:
                prior_target = (
                    self.config.lmstudio_models_dir
                    / (existing.user_repo if existing else model.import_user_repo)
                    / prior.import_alias_path.name
                )
            if (
                prior is not None
                and prior_target is not None
                and prior_target.is_file()
            ):
                completed.append(
                    SyncedArtifact(
                        role=prior.role,
                        digest=prior.digest,
                        blob_path=prior.blob_path,
                        import_alias_path=prior.import_alias_path,
                        imported_model_path=prior_target,
                        import_command=prior.import_command,
                    )
                )
                continue

            if import_mode is ImportMode.DIRECT:
                alias_path = artifact.blob_path
                alias_created = False
            else:
                try:
                    alias_path, alias_created = self._ensure_import_alias(
                        model, artifact, link_mode
                    )
                except RuntimeError as exc:
                    return self._partial_result(
                        model,
                        records,
                        completed,
                        link_mode,
                        imported_at,
                        str(exc),
                        dry_run,
                    )

            import_result: ImportResult | None = None
            import_error: Exception | None = None
            try:
                import_result = self.lmstudio.import_model(
                    str(alias_path),
                    user_repo=model.import_user_repo,
                    link_mode=link_mode,
                    dry_run=dry_run,
                )
            except (LMStudioError, subprocess.TimeoutExpired) as exc:
                import_error = exc
            finally:
                self._clean_temporary_alias(
                    alias_path,
                    alias_created,
                    link_mode,
                    import_mode,
                    dry_run,
                )

            if import_error is not None or import_result is None:
                message = (
                    str(import_error)
                    if import_error is not None
                    else "LM Studio import returned no result"
                )
                return self._partial_result(
                    model,
                    records,
                    completed,
                    link_mode,
                    imported_at,
                    message,
                    dry_run,
                )

            last_result = import_result
            completed.append(
                SyncedArtifact(
                    role=artifact.role,
                    digest=artifact.digest,
                    blob_path=artifact.blob_path,
                    import_alias_path=alias_path,
                    imported_model_path=(
                        self.config.lmstudio_models_dir
                        / model.import_user_repo
                        / alias_path.name
                    ),
                    import_command=import_result.command,
                )
            )
            if not dry_run:
                partial = self._make_record(
                    model, completed, link_mode, imported_at, vision_confirmed=False
                )
                records[model.canonical_name] = partial
                self.state.save(records)

        vision_confirmed = True
        if len(artifacts) > 1 and not dry_run:
            try:
                vision_confirmed = self._vision_bundle_is_indexed(model, completed)
            except LMStudioError as exc:
                return self._partial_result(
                    model,
                    records,
                    completed,
                    link_mode,
                    imported_at,
                    str(exc),
                    dry_run,
                )
            if not vision_confirmed:
                return self._partial_result(
                    model,
                    records,
                    completed,
                    link_mode,
                    imported_at,
                    "LM Studio did not index the projector bundle as vision-capable",
                    dry_run,
                )

        record = self._make_record(
            model, completed, link_mode, imported_at, vision_confirmed=vision_confirmed
        )
        if not dry_run:
            records[model.canonical_name] = record
            self.state.save(records)
        status = "dry-run" if dry_run else "synced"
        if already_synced and last_result is None and not dry_run:
            status = "skipped"
        return SyncResult(
            model=model,
            status=status,
            message=(
                _format_import_message(last_result)
                if last_result is not None
                else "imported artifacts already present"
            ),
            record=record,
        )

    @staticmethod
    def _model_artifacts(model: OllamaModel) -> tuple[OllamaArtifact, ...]:
        if model.artifacts:
            return model.artifacts
        if model.model_digest is None or model.blob_path is None:
            return ()
        return (
            OllamaArtifact(
                role=ArtifactRole.MODEL,
                media_type="application/vnd.ollama.image.model",
                digest=model.model_digest,
                blob_path=model.blob_path,
                declared_size=model.declared_size,
                blob_size=model.blob_size,
                gguf_valid=model.gguf_valid,
                manifest_index=0,
                issues=model.issues,
            ),
        )

    def _make_record(
        self,
        model: OllamaModel,
        artifacts: list[SyncedArtifact],
        link_mode: LinkMode,
        imported_at: datetime,
        *,
        vision_confirmed: bool,
    ) -> SyncRecord:
        return SyncRecord(
            canonical_name=model.canonical_name,
            user_repo=model.import_user_repo,
            link_mode=link_mode,
            imported_at=imported_at,
            artifacts=tuple(artifacts),
            vision_confirmed=vision_confirmed,
        )

    def _partial_result(
        self,
        model: OllamaModel,
        records: dict[str, SyncRecord],
        completed: list[SyncedArtifact],
        link_mode: LinkMode,
        imported_at: datetime,
        message: str,
        dry_run: bool,
    ) -> SyncResult:
        record = None
        if completed:
            record = self._make_record(
                model,
                completed,
                link_mode,
                imported_at,
                vision_confirmed=False,
            )
            if not dry_run:
                records[model.canonical_name] = record
                self.state.save(records)
        return SyncResult(model=model, status="error", message=message, record=record)

    def _vision_bundle_is_indexed(
        self, model: OllamaModel, artifacts: list[SyncedArtifact]
    ) -> bool:
        primary = next(
            artifact for artifact in artifacts if artifact.role is ArtifactRole.MODEL
        )
        if primary.imported_model_path is None:
            return False
        try:
            expected = primary.imported_model_path.relative_to(
                self.config.lmstudio_models_dir
            ).as_posix()
        except ValueError:
            return False
        for attempt in range(INVENTORY_ATTEMPTS):
            if any(
                item.path.replace("\\", "/").casefold() == expected.casefold()
                and item.vision
                for item in self.lmstudio.list_models()
            ):
                return True
            if attempt + 1 < INVENTORY_ATTEMPTS:
                time.sleep(INVENTORY_RETRY_DELAY)
        return False

    @staticmethod
    def _clean_temporary_alias(
        alias_path: Path,
        alias_created: bool,
        link_mode: LinkMode,
        import_mode: ImportMode,
        dry_run: bool,
    ) -> None:
        should_remove = dry_run or link_mode is LinkMode.COPY
        if not (
            should_remove
            and (alias_created or link_mode is LinkMode.COPY)
            and alias_path.exists()
            and import_mode is not ImportMode.DIRECT
        ):
            return
        try:
            alias_path.unlink()
            alias_path.parent.rmdir()
        except OSError as exc:
            logger.warning("Could not clean up import alias %s: %s", alias_path, exc)

    def _ensure_import_alias(
        self,
        model: OllamaModel,
        artifact: OllamaArtifact,
        link_mode: LinkMode,
    ) -> tuple[Path, bool]:
        staging_dir = self.config.import_staging_dir
        if link_mode is LinkMode.COPY:
            # The alias controls the human-readable LM Studio filename. Keep
            # this temporary hard link on the Ollama filesystem so copy mode
            # still works when StudioLink state lives on another volume.
            staging_dir = self.config.ollama_models_dir / ".studiolink-imports"
        return _ensure_import_alias(model, staging_dir, artifact)


def _ensure_import_alias(
    model: OllamaModel,
    staging_dir: Path,
    artifact: OllamaArtifact | None = None,
) -> tuple[Path, bool]:
    blob_path = artifact.blob_path if artifact is not None else model.blob_path
    alias_name = (
        model.artifact_import_filename(artifact)
        if artifact is not None
        else model.import_filename
    )
    alias_path = staging_dir / alias_name
    if blob_path is None:
        raise RuntimeError(f"No blob path for model {model.canonical_name}")
    if alias_path.exists():
        if same_file(alias_path, blob_path):
            logger.debug("Import alias already exists: %s", alias_path)
            return alias_path, False
        # The alias survived a re-pull but now points at the old blob content;
        # replace it so we never import stale weights.
        logger.debug("Replacing stale import alias: %s", alias_path)
        alias_path.unlink()
    staging_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.link(blob_path, alias_path)
        logger.debug("Created hard link: %s -> %s", alias_path, blob_path)
        return alias_path, True
    except OSError as exc:
        raise RuntimeError(f"Failed to create import alias (hard link): {exc}") from exc


def _format_import_message(result: ImportResult) -> str:
    for stream in (result.stdout, result.stderr):
        lines = [line.strip() for line in stream.splitlines() if line.strip()]
        if lines:
            return lines[-1]
    return "imported successfully"
