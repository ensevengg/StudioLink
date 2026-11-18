from __future__ import annotations

import errno
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from studiolink.config import StudioLinkConfig
from studiolink.files import (
    artifact_destination,
    destination_path,
    healthy_destination,
    identity,
    owns_destination,
    remove_import,
    same_file,
)
from studiolink.lmstudio_adapter import LMStudioError
from studiolink.models import (
    ArtifactRole,
    ImportMode,
    ImportResult,
    LinkMode,
    LMStudioModel,
    ModelReadiness,
    OllamaArtifact,
    OllamaModel,
    SyncedArtifact,
    SyncRecord,
    SyncResult,
    is_synced,
    model_artifacts,
)
from studiolink.ports import LMStudioPort
from studiolink.state import StateStore
from studiolink.transactions import (
    hardlinks_unsupported,
    import_transaction,
    recover_imports,
)

logger = logging.getLogger("studiolink")

INVENTORY_ATTEMPTS = 10
INVENTORY_RETRY_DELAY = 0.5


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
        *,
        keep_existing_mode: bool = False,
    ) -> list[SyncResult]:
        logger.debug(
            "Syncing %d model(s): link_mode=%s, import_mode=%s, dry_run=%s",
            len(models),
            link_mode,
            import_mode,
            dry_run,
        )
        with self.state.lock():
            return self._sync_locked(
                models, link_mode, import_mode, dry_run, keep_existing_mode
            )

    def _sync_locked(
        self,
        models: list[OllamaModel],
        link_mode: LinkMode,
        import_mode: ImportMode,
        dry_run: bool,
        keep_existing_mode: bool = False,
    ) -> list[SyncResult]:
        if not dry_run:
            recover_imports(self.config.lmstudio_models_dir)
        records = self.state.get_all_records()
        logger.debug("Loaded %d existing sync record(s)", len(records))
        results: list[SyncResult] = []
        for model in models:
            mode = link_mode
            existing = records.get(model.canonical_name)
            # Checked once per model: for a legacy copy record without a saved
            # identity this compares the whole file.
            synced = existing is not None and is_synced(
                model, existing, self.config.lmstudio_models_dir
            )
            if keep_existing_mode and existing is not None and synced:
                # Without an explicit flag, a working import keeps its mode
                # instead of being converted to whatever the probe picked.
                mode = existing.link_mode
            try:
                result = self._sync_one(
                    model, records, mode, import_mode, dry_run, synced=synced
                )
            except (OSError, RuntimeError, ValueError) as exc:
                result = SyncResult(model=model, status="error", message=str(exc))
            results.append(result)
            if (
                result.record
                and not dry_run
                and result.record != records.get(model.canonical_name)
            ):
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
        *,
        synced: bool | None = None,
    ) -> SyncResult:
        logger.debug(
            "Processing model: %s (readiness=%s)",
            model.canonical_name,
            model.readiness,
        )
        root = self.config.lmstudio_models_dir
        existing = records.get(model.canonical_name)
        sources = model_artifacts(model)
        if synced is None:
            synced = existing is not None and is_synced(model, existing, root)
        if existing is not None and synced and existing.link_mode is link_mode:
            logger.debug("Model %s is already synced, skipping", model.canonical_name)
            verified = self._verified_record(existing, sources)
            message = "imported file verified; already synced"
            if len(sources) == 1 or dry_run:
                return SyncResult(
                    model=model, status="skipped", message=message, record=verified
                )
            # LM Studio may have stopped pairing the projector since the last
            # sync, so a bundle is only skipped after its inventory is rechecked.
            return self._confirm_bundle(model, verified, "skipped", message)
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
        if model.readiness is ModelReadiness.INVALID or not sources:
            logger.debug("Model %s is invalid: %s", model.canonical_name, model.issues)
            return SyncResult(
                model=model,
                status="error",
                message="model is not ready for import: "
                + "; ".join(model.issues or ("unknown error",)),
            )
        if import_mode is ImportMode.DIRECT and len(sources) > 1:
            return SyncResult(
                model=model,
                status="error",
                message=(
                    "--direct is not supported for multi-artifact models; "
                    "projector imports require an mmproj- alias"
                ),
            )
        if dry_run:
            try:
                targets = ", ".join(
                    str(
                        destination_path(
                            root,
                            model.import_user_repo,
                            self._source_path(model, artifact, import_mode).name,
                        )
                    )
                    for artifact in sources
                )
            except ValueError as exc:
                return SyncResult(model=model, status="error", message=str(exc))
            return SyncResult(
                model=model,
                status="dry-run",
                message=f"Would sync using {link_mode}: {targets}",
            )

        completed: list[SyncedArtifact] = []
        message = "imported successfully"
        for source in sources:
            try:
                artifact, import_result = self._sync_artifact(
                    model, source, existing, records, link_mode, import_mode
                )
            except (
                LMStudioError,
                subprocess.TimeoutExpired,
                OSError,
                RuntimeError,
                ValueError,
            ) as exc:
                return self._partial_result(
                    model, existing, completed, link_mode, str(exc)
                )
            completed.append(artifact)
            message = _format_import_message(import_result)

        record = self._make_record(model, completed, link_mode)
        if len(completed) == 1:
            result = SyncResult(
                model=model, status="synced", message=message, record=record
            )
        else:
            # Confirm a bundle before reclaiming the one it replaces, so a
            # bundle LM Studio rejects never costs the previous working one.
            result = self._confirm_bundle(model, record, "synced", message)
            if result.status == "error":
                if existing is not None and not (
                    self._targets(existing) <= self._targets(record)
                ):
                    # Keep the previous import tracked; a retry adopts the
                    # new files without another copy.
                    return replace(result, record=None)
                return result
            record = result.record or record
        if existing is not None:
            try:
                self._reclaim(existing, record, records)
            except (OSError, RuntimeError, ValueError) as exc:
                # Keep the old record so its cleanup is retried; the new import
                # is verified and will be adopted without another copy.
                return SyncResult(model=model, status="error", message=str(exc))
        return result

    def _sync_artifact(
        self,
        model: OllamaModel,
        source: OllamaArtifact,
        existing: SyncRecord | None,
        records: dict[str, SyncRecord],
        link_mode: LinkMode,
        import_mode: ImportMode,
    ) -> tuple[SyncedArtifact, ImportResult]:
        blob, digest = source.blob_path, source.digest
        if blob is None or digest is None:
            raise RuntimeError(f"{source.role.value} artifact is not importable")
        if import_mode is ImportMode.DIRECT:
            alias_path = blob
            alias_created = False
            logger.debug("Using Ollama blob directly: %s", alias_path)
        else:
            alias_path, alias_created = self._ensure_import_alias(
                model, source, link_mode
            )
            logger.debug(
                "Importing %s via LM Studio: %s (mode=%s)",
                source.role,
                alias_path,
                link_mode,
            )

        root = self.config.lmstudio_models_dir
        target = destination_path(root, model.import_user_repo, alias_path.name)
        prior = None
        if existing is not None:
            prior = next(
                (
                    artifact
                    for artifact in existing.artifacts
                    if artifact_destination(existing, artifact, root) == target
                ),
                None,
            )
        import_result = ImportResult(
            command=(), stdout="existing import verified", stderr="", return_code=0
        )
        target_artifact = (
            prior
            if existing is not None
            and prior is not None
            and prior.digest == digest
            and existing.link_mode is link_mode
            else None
        )
        success = False
        try:
            if not healthy_destination(target, blob, link_mode, target_artifact):
                if target.exists() or target.is_symlink():
                    managed = (
                        existing is not None
                        and prior is not None
                        and owns_destination(existing, prior, target)
                    )
                    # Lost state can be rebuilt from a verified import even
                    # when the requested mode differs from its current mode.
                    recoverable = existing is None and any(
                        healthy_destination(target, blob, candidate)
                        for candidate in LinkMode
                    )
                    if not (managed or recoverable):
                        raise RuntimeError(
                            f"Import destination already exists and is not verified: {target}"
                        )
                    # Build beside the destination and replace atomically. A
                    # failed conversion leaves the old import usable.
                    self._replace_destination(
                        alias_path, target, link_mode, blob, digest, records
                    )
                    import_result = ImportResult(
                        command=(),
                        stdout=f"Replaced managed import: {target}",
                        stderr="",
                        return_code=0,
                    )
                elif link_mode is LinkMode.COPY and self._reuse_copy(
                    blob, digest, records, target
                ):
                    import_result = ImportResult(
                        command=(),
                        stdout="Reused existing copy of identical weights",
                        stderr="",
                        return_code=0,
                    )
                else:
                    import_result = self.lmstudio.import_model(
                        str(alias_path.absolute()),
                        user_repo=model.import_user_repo,
                        link_mode=link_mode,
                        dry_run=False,
                    )
                if not healthy_destination(
                    target, blob, link_mode, compare_contents=False
                ):
                    raise RuntimeError(
                        f"LM Studio did not create the expected {link_mode} import at {target}; "
                        "check that STUDIOLINK_LMSTUDIO_MODELS_DIR matches LM Studio's models directory"
                    )
            success = True
        finally:
            if import_mode is not ImportMode.DIRECT and (
                (alias_created and not success) or link_mode is LinkMode.COPY
            ):
                # Preserve a symlink target if import succeeded but a later cleanup
                # failed. A subsequent sync can adopt that verified destination.
                if not (target.is_symlink() and same_file(target, alias_path)):
                    alias_path.unlink(missing_ok=True)
                    if link_mode is LinkMode.COPY:
                        try:
                            alias_path.parent.rmdir()
                        except OSError:
                            pass

        artifact = SyncedArtifact(
            role=source.role,
            digest=digest,
            blob_path=blob,
            import_alias_path=alias_path,
            imported_model_path=target,
            import_command=import_result.command
            or (prior.import_command if prior is not None else ()),
            imported_identity=identity(target),
        )
        return artifact, import_result

    def _source_path(
        self, model: OllamaModel, artifact: OllamaArtifact, import_mode: ImportMode
    ) -> Path:
        if import_mode is ImportMode.DIRECT and artifact.blob_path is not None:
            return artifact.blob_path
        return self.config.import_staging_dir / model.artifact_import_filename(artifact)

    def _verified_record(
        self, record: SyncRecord, sources: tuple[OllamaArtifact, ...]
    ) -> SyncRecord:
        artifacts = []
        for artifact, source in zip(record.artifacts, sources):
            target = artifact_destination(
                record, artifact, self.config.lmstudio_models_dir
            )
            artifacts.append(
                replace(
                    artifact,
                    imported_model_path=target,
                    imported_identity=identity(target),
                    blob_path=source.blob_path or artifact.blob_path,
                )
            )
        return replace(record, artifacts=tuple(artifacts))

    def _make_record(
        self,
        model: OllamaModel,
        artifacts: list[SyncedArtifact],
        link_mode: LinkMode,
    ) -> SyncRecord:
        return SyncRecord(
            canonical_name=model.canonical_name,
            user_repo=model.import_user_repo,
            link_mode=link_mode,
            imported_at=datetime.now(tz=timezone.utc),
            artifacts=tuple(artifacts),
            # Bundles are confirmed against LM Studio's inventory afterwards.
            vision_confirmed=not any(
                artifact.role is ArtifactRole.PROJECTOR for artifact in artifacts
            ),
        )

    def _partial_result(
        self,
        model: OllamaModel,
        existing: SyncRecord | None,
        completed: list[SyncedArtifact],
        link_mode: LinkMode,
        message: str,
    ) -> SyncResult:
        record = None
        # Model artifacts import first, so a partial record always has a
        # primary. Saving it lets a retry resume without another import.
        if any(artifact.role is ArtifactRole.MODEL for artifact in completed):
            partial = self._make_record(model, completed, link_mode)
            # A failed re-pull must not drop the previous import from state:
            # it stays tracked until a complete replacement can reclaim it.
            if existing is None or self._targets(existing) <= self._targets(partial):
                record = partial
        return SyncResult(model=model, status="error", message=message, record=record)

    def _targets(self, record: SyncRecord) -> set[Path]:
        root = self.config.lmstudio_models_dir
        return {
            artifact_destination(record, artifact, root)
            for artifact in record.artifacts
        }

    def _reclaim(
        self,
        existing: SyncRecord,
        record: SyncRecord,
        records: dict[str, SyncRecord],
    ) -> None:
        """Remove imports and aliases the new record no longer uses."""
        root = self.config.lmstudio_models_dir
        keep = self._targets(record)
        keep.update(artifact.import_alias_path for artifact in record.artifacts)
        for name, other in records.items():
            if name != record.canonical_name:
                keep.update(self._targets(other))
                keep.update(artifact.import_alias_path for artifact in other.artifacts)
        remove_import(existing, root, self.config.import_staging_dir, keep=keep)

    def _confirm_bundle(
        self, model: OllamaModel, record: SyncRecord, status: str, message: str
    ) -> SyncResult:
        try:
            confirmed = self._vision_bundle_is_indexed(record)
        except LMStudioError as exc:
            return SyncResult(
                model=model,
                status="error",
                message=str(exc),
                record=replace(record, vision_confirmed=False),
            )
        if not confirmed:
            return SyncResult(
                model=model,
                status="error",
                message="LM Studio did not index the projector bundle as vision-capable",
                record=replace(record, vision_confirmed=False),
            )
        return SyncResult(
            model=model,
            status=status,
            message=message,
            record=replace(record, vision_confirmed=True),
        )

    def _vision_bundle_is_indexed(self, record: SyncRecord) -> bool:
        for attempt in range(INVENTORY_ATTEMPTS):
            if bundle_is_indexed(
                record, self.config.lmstudio_models_dir, self.lmstudio.list_models()
            ):
                return True
            if attempt + 1 < INVENTORY_ATTEMPTS:
                time.sleep(INVENTORY_RETRY_DELAY)
        return False

    def _reuse_copy(
        self,
        blob: Path,
        digest: str,
        records: dict[str, SyncRecord],
        target: Path,
    ) -> bool:
        root = self.config.lmstudio_models_dir
        for record in records.values():
            if record.link_mode is not LinkMode.COPY:
                continue
            for artifact in record.artifacts:
                if artifact.digest != digest:
                    continue
                source = artifact_destination(record, artifact, root)
                if source == target or not healthy_destination(
                    source, blob, LinkMode.COPY, artifact
                ):
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source, target)
                except OSError as exc:
                    if exc.errno == errno.EXDEV or hardlinks_unsupported(exc):
                        # The caller falls back to an ordinary copy.
                        continue
                    raise
                return True
        return False

    def _replace_destination(
        self,
        source: Path,
        target: Path,
        mode: LinkMode,
        blob: Path,
        digest: str,
        records: dict[str, SyncRecord],
    ) -> None:
        with import_transaction(
            self.config.lmstudio_models_dir, target.name
        ) as temporary:
            if mode is LinkMode.HARD_LINK:
                os.link(source.resolve(), temporary)
            elif mode is LinkMode.SYMBOLIC_LINK:
                temporary.symlink_to(source.absolute())
            elif not self._reuse_copy(blob, digest, records, temporary):
                shutil.copyfile(source, temporary)
            os.replace(temporary, target)

    def _ensure_import_alias(
        self, model: OllamaModel, artifact: OllamaArtifact, link_mode: LinkMode
    ) -> tuple[Path, bool]:
        staging = self.config.import_staging_dir
        if link_mode is LinkMode.COPY:
            # The alias controls the human-readable LM Studio filename. Keep
            # this temporary hard link on the Ollama filesystem so copy mode
            # still works when StudioLink state lives on another volume.
            beside_blobs = self.config.ollama_models_dir / ".studiolink-imports"
            try:
                return _ensure_import_alias(model, beside_blobs, link_mode, artifact)
            except RuntimeError as exc:
                # A system-service store (owned by the "ollama" user) is
                # read-only to us; a symlink alias in our own staging works,
                # since LM Studio copies the file it resolves to.
                logger.debug(
                    "Copy alias beside blobs failed (%s); using %s", exc, staging
                )
        return _ensure_import_alias(model, staging, link_mode, artifact)


def bundle_is_indexed(
    record: SyncRecord, root: Path, inventory: tuple[LMStudioModel, ...]
) -> bool:
    """Check that LM Studio paired the record's projector with its model."""
    try:
        expected = (
            artifact_destination(record, record.primary_artifact, root)
            .relative_to(root.absolute())
            .as_posix()
        )
    except ValueError:
        return False
    return any(
        item.path.replace("\\", "/").casefold() == expected.casefold() and item.vision
        for item in inventory
    )


def _ensure_import_alias(
    model: OllamaModel,
    staging_dir: Path,
    link_mode: LinkMode = LinkMode.HARD_LINK,
    artifact: OllamaArtifact | None = None,
) -> tuple[Path, bool]:
    blob_path = artifact.blob_path if artifact is not None else model.blob_path
    alias_path = staging_dir / (
        model.artifact_import_filename(artifact)
        if artifact is not None
        else model.import_filename
    )
    if blob_path is None:
        raise RuntimeError(f"No blob path for model {model.canonical_name}")
    present = alias_path.exists() or alias_path.is_symlink()
    if (
        present
        and same_file(alias_path, blob_path)
        and (link_mode is not LinkMode.HARD_LINK or not alias_path.is_symlink())
    ):
        return alias_path, False
    temporary = alias_path
    try:
        staging_dir.mkdir(parents=True, exist_ok=True)
        if present:
            # Replace a stale alias atomically so a failure keeps the old link.
            fd, name = tempfile.mkstemp(prefix=".studiolink-", dir=staging_dir)
            os.close(fd)
            temporary = Path(name)
            temporary.unlink()
        try:
            os.link(blob_path, temporary)
        except OSError as exc:
            if not (exc.errno == errno.EXDEV or hardlinks_unsupported(exc)):
                raise
            if link_mode is LinkMode.HARD_LINK:
                raise RuntimeError(
                    f"Failed to create import alias: {exc}. Hard links are not "
                    "possible here (another filesystem or btrfs subvolume, or a "
                    "blob owned by another user under fs.protected_hardlinks); "
                    "use --symbolic-link or --copy, or omit the flag to choose "
                    "automatically"
                ) from exc
            # Stage across volumes, or around fs.protected_hardlinks, without
            # writing another full copy.
            temporary.symlink_to(blob_path.resolve())
        if present:
            os.replace(temporary, alias_path)
        return alias_path, True
    except OSError as exc:
        raise RuntimeError(f"Failed to create import alias: {exc}") from exc
    finally:
        if temporary != alias_path:
            temporary.unlink(missing_ok=True)


def _format_import_message(result: ImportResult) -> str:
    for stream in (result.stdout, result.stderr):
        lines = [line.strip() for line in stream.splitlines() if line.strip()]
        if lines:
            return lines[-1]
    return "imported successfully"
