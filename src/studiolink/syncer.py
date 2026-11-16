from __future__ import annotations

import errno
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from studiolink.config import StudioLinkConfig
from studiolink.files import (
    destination_path,
    healthy_destination,
    identity,
    owns_destination,
    record_destination,
    remove_import,
    same_file,
    symlink_references,
)
from studiolink.lmstudio_adapter import LMStudioError
from studiolink.models import (
    ImportMode,
    ImportResult,
    LinkMode,
    ModelReadiness,
    OllamaModel,
    SyncRecord,
    SyncResult,
    is_synced,
)
from studiolink.ports import LMStudioPort
from studiolink.state import StateStore
from studiolink.transactions import import_transaction, recover_imports

logger = logging.getLogger("studiolink")


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
        with self.state.lock():
            return self._sync_locked(models, link_mode, import_mode, dry_run)

    def _sync_locked(
        self,
        models: list[OllamaModel],
        link_mode: LinkMode,
        import_mode: ImportMode,
        dry_run: bool,
    ) -> list[SyncResult]:
        if not dry_run:
            recover_imports(self.config.lmstudio_models_dir)
        records = self.state.get_all_records()
        logger.debug("Loaded %d existing sync record(s)", len(records))
        results: list[SyncResult] = []
        for model in models:
            try:
                result = self._sync_one(model, records, link_mode, import_mode, dry_run)
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
    ) -> SyncResult:
        logger.debug(
            "Processing model: %s (readiness=%s)",
            model.canonical_name,
            model.readiness,
        )
        existing = records.get(model.canonical_name)
        if (
            existing is not None
            and is_synced(model, existing, self.config.lmstudio_models_dir)
            and existing.link_mode is link_mode
        ):
            logger.debug("Model %s is already synced, skipping", model.canonical_name)
            verified_target = record_destination(
                existing, self.config.lmstudio_models_dir
            )
            return SyncResult(
                model=model,
                status="skipped",
                message="imported file verified; already synced",
                record=replace(
                    existing,
                    imported_model_path=verified_target,
                    imported_identity=identity(verified_target),
                    blob_path=model.blob_path,
                ),
            )
        if model.readiness is ModelReadiness.STALE:
            logger.debug("Model %s is stale (blob missing)", model.canonical_name)
            return SyncResult(
                model=model,
                status="error",
                message=(
                    "model blob is missing from the Ollama blob store; "
                    f"run `ollama pull {model.canonical_name}` to restore it"
                ),
            )
        if (
            model.readiness is ModelReadiness.INVALID
            or model.blob_path is None
            or model.model_digest is None
        ):
            logger.debug("Model %s is invalid: %s", model.canonical_name, model.issues)
            return SyncResult(
                model=model,
                status="error",
                message="model is not ready for import: "
                + "; ".join(model.issues or ("unknown error",)),
            )
        if dry_run:
            source = (
                model.blob_path
                if import_mode is ImportMode.DIRECT
                else self.config.import_staging_dir / model.import_filename
            )
            target = destination_path(
                self.config.lmstudio_models_dir, model.user_repo, source.name
            )
            return SyncResult(
                model=model,
                status="dry-run",
                message=f"Would sync using {link_mode}: {target}",
            )
        if import_mode is ImportMode.DIRECT:
            alias_path = model.blob_path
            alias_created = False
            logger.debug("Using Ollama blob directly: %s", alias_path)
        else:
            try:
                alias_path, alias_created = self._ensure_import_alias(model, link_mode)
                logger.debug(
                    "Importing model via LM Studio: %s (mode=%s, dry_run=%s)",
                    alias_path,
                    link_mode,
                    dry_run,
                )
            except RuntimeError as exc:
                logger.debug("Failed to create import alias: %s", exc)
                return SyncResult(
                    model=model,
                    status="error",
                    message=str(exc),
                )

        target = destination_path(
            self.config.lmstudio_models_dir, model.user_repo, alias_path.name
        )
        import_result = ImportResult(
            command=(), stdout="existing import verified", stderr="", return_code=0
        )
        target_record = (
            existing
            if existing
            and existing.digest == model.model_digest
            and existing.link_mode is link_mode
            and record_destination(existing, self.config.lmstudio_models_dir) == target
            else None
        )
        success = False
        try:
            if not healthy_destination(
                target, model.blob_path, link_mode, target_record
            ):
                if target.exists() or target.is_symlink():
                    managed = existing is not None and (
                        record_destination(existing, self.config.lmstudio_models_dir)
                        == target
                        and owns_destination(existing, target)
                    )
                    # Lost state can be rebuilt from a verified import even
                    # when the requested mode differs from its current mode.
                    recoverable = existing is None and any(
                        healthy_destination(target, model.blob_path, candidate)
                        for candidate in LinkMode
                    )
                    if not (managed or recoverable):
                        raise RuntimeError(
                            f"Import destination already exists and is not verified: {target}"
                        )
                    # Build beside the destination and replace atomically. A
                    # failed conversion leaves the old import usable.
                    self._replace_destination(
                        alias_path, target, link_mode, model, records
                    )
                    import_result = ImportResult(
                        command=(),
                        stdout=f"Replaced managed import: {target}",
                        stderr="",
                        return_code=0,
                    )
                else:
                    import_result = self.lmstudio.import_model(
                        str(alias_path.absolute()),
                        user_repo=model.user_repo,
                        link_mode=link_mode,
                        dry_run=False,
                    )
                if not healthy_destination(target, model.blob_path, link_mode):
                    raise RuntimeError(
                        f"LM Studio did not create the expected {link_mode} import at {target}; "
                        "check that STUDIOLINK_LMSTUDIO_MODELS_DIR matches LM Studio's models directory"
                    )
            if existing is not None:
                keep = {target, alias_path}
                for name, record in records.items():
                    if name != model.canonical_name:
                        keep.add(record.import_alias_path)
                        keep.add(
                            record_destination(record, self.config.lmstudio_models_dir)
                        )
                old_target = record_destination(
                    existing, self.config.lmstudio_models_dir
                )
                if old_target != target:
                    remove_import(
                        existing,
                        self.config.lmstudio_models_dir,
                        self.config.import_staging_dir,
                        keep=keep,
                    )
                elif (
                    existing.import_alias_path not in keep
                    and existing.import_alias_path.parent.resolve()
                    == self.config.import_staging_dir.resolve()
                ):
                    if existing.import_alias_path.absolute() not in symlink_references(
                        self.config.lmstudio_models_dir
                    ):
                        existing.import_alias_path.unlink(missing_ok=True)
            success = True
        except (
            LMStudioError,
            subprocess.TimeoutExpired,
            OSError,
            RuntimeError,
            ValueError,
        ) as exc:
            return SyncResult(model=model, status="error", message=str(exc))
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

        record = SyncRecord(
            canonical_name=model.canonical_name,
            digest=model.model_digest,
            blob_path=model.blob_path,
            import_alias_path=alias_path,
            user_repo=model.user_repo,
            link_mode=link_mode,
            imported_at=datetime.now(tz=timezone.utc),
            import_command=import_result.command,
            imported_model_path=target,
            imported_identity=identity(target),
        )
        return SyncResult(
            model=model,
            status="synced",
            message=_format_import_message(import_result),
            record=record,
        )

    def _replace_destination(
        self,
        source: Path,
        target: Path,
        mode: LinkMode,
        model: OllamaModel,
        records: dict[str, SyncRecord],
    ) -> None:
        with import_transaction(
            self.config.lmstudio_models_dir, target.name
        ) as temporary:
            if mode is LinkMode.HARD_LINK:
                os.link(source.resolve(), temporary)
            elif mode is LinkMode.SYMBOLIC_LINK:
                temporary.symlink_to(source.absolute())
            else:
                shutil.copyfile(source, temporary)
            os.replace(temporary, target)

    def _ensure_import_alias(
        self, model: OllamaModel, link_mode: LinkMode
    ) -> tuple[Path, bool]:
        staging = self.config.import_staging_dir
        if link_mode is LinkMode.COPY:
            staging = self.config.ollama_models_dir / ".studiolink-imports"
        return _ensure_import_alias(model, staging, link_mode)


def _ensure_import_alias(
    model: OllamaModel, staging_dir: Path, link_mode: LinkMode = LinkMode.HARD_LINK
) -> tuple[Path, bool]:
    alias_path = staging_dir / model.import_filename
    if model.blob_path is None:
        raise RuntimeError(f"No blob path for model {model.canonical_name}")
    present = alias_path.exists() or alias_path.is_symlink()
    if (
        present
        and same_file(alias_path, model.blob_path)
        and (link_mode is not LinkMode.HARD_LINK or not alias_path.is_symlink())
    ):
        return alias_path, False
    staging_dir.mkdir(parents=True, exist_ok=True)
    temporary = alias_path
    if present:
        fd, name = tempfile.mkstemp(prefix=".studiolink-", dir=staging_dir)
        os.close(fd)
        temporary = Path(name)
        temporary.unlink()
    try:
        try:
            os.link(model.blob_path, temporary)
        except OSError as exc:
            if exc.errno != errno.EXDEV or link_mode is LinkMode.HARD_LINK:
                raise
            # Stage across volumes without writing another full copy.
            temporary.symlink_to(model.blob_path.resolve())
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
