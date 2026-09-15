from __future__ import annotations

import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from studiolink.config import StudioLinkConfig
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

logger = logging.getLogger("studiolink")


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
        if is_synced(model, existing, expected_target=expected_target):
            logger.debug("Model %s is already synced, skipping", model.canonical_name)
            return SyncResult(
                model=model,
                status="skipped",
                message="already synced according to StudioLink state",
                record=existing,
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

        import_error: Exception | None = None
        import_result: ImportResult | None = None
        try:
            import_result = self.lmstudio.import_model(
                str(alias_path),
                user_repo=model.user_repo,
                link_mode=link_mode,
                dry_run=dry_run,
            )
            logger.debug("LM Studio import completed: %s", import_result.return_code)
        except (LMStudioError, subprocess.TimeoutExpired) as exc:
            # One failing model must not abort the rest of the batch.
            logger.debug("LM Studio import failed: %s", exc)
            import_error = exc
        finally:
            should_remove_alias = dry_run or link_mode is LinkMode.COPY
            if (
                should_remove_alias
                and (alias_created or link_mode is LinkMode.COPY)
                and alias_path.exists()
                and import_mode is not ImportMode.DIRECT
            ):
                logger.debug("Cleaning up temporary import alias: %s", alias_path)
                try:
                    alias_path.unlink()
                    alias_path.parent.rmdir()
                except OSError as exc:
                    logger.warning(
                        "Could not clean up import alias %s: %s", alias_path, exc
                    )

        if import_error is not None:
            return SyncResult(model=model, status="error", message=str(import_error))
        if import_result is None:
            return SyncResult(
                model=model,
                status="error",
                message="LM Studio import returned no result",
            )

        record = SyncRecord(
            canonical_name=model.canonical_name,
            digest=model.model_digest,
            blob_path=model.blob_path,
            import_alias_path=alias_path,
            user_repo=model.user_repo,
            link_mode=link_mode,
            imported_at=datetime.now(tz=timezone.utc),
            imported_model_path=(
                self.config.lmstudio_models_dir / model.user_repo / alias_path.name
            ),
            import_command=import_result.command,
        )
        return SyncResult(
            model=model,
            status="dry-run" if dry_run else "synced",
            message=_format_import_message(import_result),
            record=record,
        )

    def _ensure_import_alias(
        self, model: OllamaModel, link_mode: LinkMode
    ) -> tuple[Path, bool]:
        staging_dir = self.config.import_staging_dir
        if link_mode is LinkMode.COPY:
            # The alias controls the human-readable LM Studio filename. Keep
            # this temporary hard link on the Ollama filesystem so copy mode
            # still works when StudioLink state lives on another volume.
            staging_dir = self.config.ollama_models_dir / ".studiolink-imports"
        return _ensure_import_alias(model, staging_dir)


def _ensure_import_alias(model: OllamaModel, staging_dir: Path) -> tuple[Path, bool]:
    alias_path = staging_dir / model.import_filename
    if model.blob_path is None:
        raise RuntimeError(f"No blob path for model {model.canonical_name}")
    if alias_path.exists():
        if same_file(alias_path, model.blob_path):
            logger.debug("Import alias already exists: %s", alias_path)
            return alias_path, False
        # The alias survived a re-pull but now points at the old blob content;
        # replace it so we never import stale weights.
        logger.debug("Replacing stale import alias: %s", alias_path)
        alias_path.unlink()
    staging_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.link(model.blob_path, alias_path)
        logger.debug("Created hard link: %s -> %s", alias_path, model.blob_path)
        return alias_path, True
    except OSError as exc:
        raise RuntimeError(f"Failed to create import alias (hard link): {exc}") from exc


def _format_import_message(result: ImportResult) -> str:
    for stream in (result.stdout, result.stderr):
        lines = [line.strip() for line in stream.splitlines() if line.strip()]
        if lines:
            return lines[-1]
    return "imported successfully"
