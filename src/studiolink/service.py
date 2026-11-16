from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from studiolink.config import StudioLinkConfig
from studiolink.files import owns_destination, record_destination, symlink_references
from studiolink.lmstudio_adapter import LMStudioAdapter
from studiolink.models import (
    DoctorCheck,
    ImportMode,
    LinkMode,
    ModelReadiness,
    OllamaModel,
    OllamaScanReport,
    PruneReport,
    PruneResult,
    SyncRecord,
    SyncResult,
    is_synced,
)
from studiolink.ollama_adapter import OllamaAdapter
from studiolink.state import StateStore
from studiolink.syncer import Syncer, same_file
from studiolink.transactions import recover_imports

if TYPE_CHECKING:
    from studiolink.ports import LMStudioPort, OllamaPort

logger = logging.getLogger("studiolink")


@dataclass(slots=True, frozen=True)
class StatusEntry:
    model: OllamaModel
    synced: bool
    sync_record: SyncRecord | None

    @property
    def display_status(self) -> str:
        if self.synced:
            return "synced"
        if self.model.readiness is ModelReadiness.STALE:
            return "stale"
        if self.model.readiness is ModelReadiness.INVALID:
            return "invalid"
        return "pending"


class StudioLinkService:
    def __init__(
        self,
        config: StudioLinkConfig | None = None,
        ollama: "OllamaPort | None" = None,
        lmstudio: "LMStudioPort | None" = None,
    ) -> None:
        self.config = config or StudioLinkConfig.from_env()
        self.ollama = ollama if ollama else OllamaAdapter(self.config)
        self.lmstudio = lmstudio if lmstudio else LMStudioAdapter(self.config)
        self.state = StateStore(self.config.state_file)

    def scan(self) -> list[OllamaModel]:
        report = self.scan_report()
        return list(report.models)

    def scan_report(self) -> OllamaScanReport:
        logger.debug(
            "Scanning Ollama manifests at %s", self.config.ollama_manifests_dir
        )
        report = self.ollama.scan_report()
        logger.debug("Discovered %d model(s)", len(report.models))
        return report

    def status(self) -> list[StatusEntry]:
        models = self.scan()
        records = self.state.get_all_records()
        return [
            StatusEntry(
                model=model,
                synced=is_synced(
                    model,
                    records.get(model.canonical_name),
                    self.config.lmstudio_models_dir,
                ),
                sync_record=records.get(model.canonical_name),
            )
            for model in models
        ]

    def sync(
        self,
        *,
        model_names: list[str] | None = None,
        sync_all: bool = False,
        link_mode: LinkMode | None = None,
        import_mode: ImportMode = ImportMode.ALIAS,
        dry_run: bool = False,
    ) -> list[SyncResult]:
        logger.debug(
            "Starting sync: sync_all=%s, link_mode=%s, import_mode=%s, dry_run=%s",
            sync_all,
            link_mode,
            import_mode,
            dry_run,
        )

        discovered = self.scan()

        if sync_all:
            selected = discovered
            logger.debug("Syncing all %d discovered model(s)", len(selected))
        else:
            selected = self._select_models(discovered, model_names or [])
            logger.debug(
                "Syncing selected model(s): %s", [m.canonical_name for m in selected]
            )

        mode = link_mode or self.config.default_link_mode
        syncer = Syncer(self.config, self.lmstudio, self.state)
        return syncer.sync(selected, mode, import_mode, dry_run)

    def prune(self, *, dry_run: bool = False) -> PruneReport:
        """Remove obsolete managed destinations as well as their staging aliases."""
        with self.state.lock():
            return self._prune_locked(dry_run=dry_run)

    def _prune_locked(self, *, dry_run: bool) -> PruneReport:
        scan = self.scan_report()
        if not scan.source_available or not self.config.ollama_blobs_dir.is_dir():
            details = "; ".join(scan.errors) or "blobs directory is unavailable"
            raise RuntimeError(f"Ollama library could not be scanned: {details}")
        if not scan.complete:
            details = "; ".join(scan.errors) or "unknown scan error"
            raise RuntimeError(f"Ollama manifest scan was incomplete: {details}")
        if not self.config.lmstudio_models_dir.is_dir():
            raise RuntimeError(
                "LM Studio models directory is unavailable; refusing to prune"
            )
        models = list(scan.models)
        current = {m.canonical_name: m for m in models}
        records = self.state.get_all_records()
        stale = {
            name: record
            for name, record in records.items()
            if name not in current
            or (
                current[name].readiness is ModelReadiness.READY
                and record.digest != current[name].model_digest
            )
        }
        protected_aliases = {
            record.import_alias_path
            for name, record in records.items()
            if name not in stale
        }
        protected_targets = {
            record_destination(record, self.config.lmstudio_models_dir)
            for name, record in records.items()
            if name not in stale
        }
        recovered = recover_imports(self.config.lmstudio_models_dir, dry_run=dry_run)
        references = symlink_references(self.config.lmstudio_models_dir)
        expected = {m.import_filename: m for m in models}
        imports: list[PruneResult] = list(recovered)
        aliases: list[PruneResult] = []
        removed_names: list[str] = []
        seen: set = set()
        # Simulate link-count changes in dry runs as well, so an inode shared by
        # several doomed paths is counted exactly once.
        removed_links: dict[tuple[int, int], int] = {}

        def remove(path, reason, results):
            if results is aliases and any(
                ref.is_symlink()
                for ref in references.get(path.absolute(), ())
                if ref not in seen
            ):
                return False
            if path in seen:
                return True
            try:
                stat = path.lstat()
            except FileNotFoundError:
                return True
            is_link = path.is_symlink()
            key = (stat.st_dev, stat.st_ino)
            remaining = stat.st_nlink - (removed_links.get(key, 0) if dry_run else 0)
            removed = False
            error = None
            if not dry_run:
                try:
                    path.unlink()
                    removed = True
                except OSError as exc:
                    error = exc
                    logger.warning("Could not remove %s: %s", path, exc)
            if dry_run or removed:
                removed_links[key] = removed_links.get(key, 0) + 1
                seen.add(path)
            results.append(
                PruneResult(
                    path=path,
                    size=stat.st_size,
                    would_free=not is_link and remaining == 1,
                    reason=reason if error is None else str(error),
                    removed=removed,
                )
            )
            return dry_run or removed

        for name, record in stale.items():
            target = record_destination(record, self.config.lmstudio_models_dir)
            if (
                record.link_mode is LinkMode.SYMBOLIC_LINK
                and record.imported_model_path is None
            ):
                # A legacy record does not identify which symlink depends on this alias.
                protected_aliases.add(record.import_alias_path)
                continue
            if target not in protected_targets and (
                target.exists() or target.is_symlink()
            ):
                if not owns_destination(record, target):
                    logger.warning("Keeping changed or unverified import: %s", target)
                    protected_aliases.add(record.import_alias_path)
                    continue
                if not remove(
                    target, "model removed from Ollama or superseded", imports
                ):
                    protected_aliases.add(record.import_alias_path)
                    continue
            alias = record.import_alias_path
            if (
                alias not in protected_aliases
                and alias.parent.resolve() == self.config.import_staging_dir.resolve()
            ):
                if not remove(alias, "obsolete import alias", aliases):
                    continue
            removed_names.append(name)

        for staging in (
            self.config.import_staging_dir,
            self.config.ollama_models_dir / ".studiolink-imports",
        ):
            if staging.is_dir():
                for path in sorted(staging.iterdir()):
                    if (
                        path in protected_aliases
                        or path in seen
                        or not (path.is_file() or path.is_symlink())
                    ):
                        continue
                    model = expected.get(path.name)
                    # An invalid/incomplete manifest isn't proof its alias is obsolete.
                    if (
                        staging == self.config.import_staging_dir
                        and model is not None
                        and (
                            model.readiness is not ModelReadiness.READY
                            or (
                                model.blob_path is not None
                                and same_file(path, model.blob_path)
                            )
                        )
                    ):
                        continue
                    reason = (
                        "alias no longer points at the current model blob"
                        if model
                        else "model no longer present in Ollama (or was re-pulled)"
                    )
                    remove(path, reason, aliases)

        if not dry_run and removed_names:
            for name in removed_names:
                records.pop(name)
            self.state.save(records)
        return PruneReport(
            dry_run=dry_run,
            aliases=tuple(aliases),
            imports=tuple(imports),
            records_removed=tuple(sorted(removed_names)),
        )

    def doctor(self) -> list[DoctorCheck]:
        checks = [
            DoctorCheck(
                "ollama executable",
                self.config.ollama_exe.exists(),
                str(self.config.ollama_exe),
            ),
            DoctorCheck(
                "lm studio executable",
                self.config.lms_exe.exists(),
                str(self.config.lms_exe),
            ),
            DoctorCheck(
                "ollama manifests dir",
                self.config.ollama_manifests_dir.exists(),
                str(self.config.ollama_manifests_dir),
            ),
            DoctorCheck(
                "ollama blobs dir",
                self.config.ollama_blobs_dir.exists(),
                str(self.config.ollama_blobs_dir),
            ),
            DoctorCheck(
                "lm studio models dir",
                self.config.lmstudio_models_dir.exists(),
                str(self.config.lmstudio_models_dir),
            ),
            self._volume_compatibility_check(),
        ]

        try:
            version = self.lmstudio.get_version()
            logger.debug("Doctor: LM Studio version = %s", version)
            checks.append(
                DoctorCheck(
                    "lm studio cli version",
                    version is not None,
                    version or "no version output",
                )
            )
        except Exception as exc:
            logger.debug("Doctor: LM Studio version check failed: %s", exc)
            checks.append(DoctorCheck("lm studio cli version", False, str(exc)))

        try:
            capabilities = self.lmstudio.get_import_capabilities()
            logger.debug("Doctor: LM Studio import capabilities = %s", capabilities)
            checks.append(
                DoctorCheck(
                    "lm studio import capabilities",
                    bool(capabilities),
                    ", ".join(sorted(c.value for c in capabilities)) or "none",
                )
            )
        except Exception as exc:
            logger.debug("Doctor: LM Studio capabilities check failed: %s", exc)
            checks.append(DoctorCheck("lm studio import capabilities", False, str(exc)))

        models = self.scan()
        checks.append(
            DoctorCheck(
                "discovered ollama models",
                True,
                f"{len(models)} model(s)",
            )
        )

        stale_models = [m for m in models if m.readiness is ModelReadiness.STALE]
        checks.append(
            DoctorCheck(
                "ollama blob presence",
                not stale_models,
                ", ".join(m.canonical_name for m in stale_models)
                if stale_models
                else "all present",
            )
        )

        valid_gguf = [m for m in models if m.gguf_valid]
        checks.append(
            DoctorCheck(
                "gguf header validation",
                len(valid_gguf) == len(models),
                f"{len(valid_gguf)}/{len(models)} passed validation",
            )
        )

        staging = self.config.import_staging_dir
        if staging.exists():
            alias_check = DoctorCheck(
                "import alias directory", staging.is_dir(), str(staging)
            )
        else:
            parent = staging.parent
            while not parent.exists() and parent != parent.parent:
                parent = parent.parent
            creatable = parent.is_dir() and os.access(parent, os.W_OK)
            details = (
                f"{staging} (will be created)"
                if creatable
                else f"{staging} (parent is not writable)"
            )
            alias_check = DoctorCheck("import alias directory", creatable, details)
        checks.append(alias_check)

        return checks

    def _volume_compatibility_check(self) -> DoctorCheck:
        """Check that hard links can span blobs -> staging -> LM Studio models.

        Compares filesystem device ids, which works on both Windows (volume
        serial) and POSIX (st_dev); drive letters alone are meaningless on
        Linux.
        """
        targets = [
            ("ollama blobs", self.config.ollama_blobs_dir),
            ("import staging", self.config.import_staging_dir),
            ("lm studio models", self.config.lmstudio_models_dir),
        ]
        devices: dict[str, int] = {}
        for label, path in targets:
            if not path.exists():
                return DoctorCheck(
                    "hard-link volume compatibility",
                    True,
                    f"skipped: {label} directory does not exist yet ({path})",
                )
            devices[label] = os.stat(path).st_dev
        ok = len(set(devices.values())) == 1
        details = ", ".join(
            f"{label}: device {device}" for label, device in devices.items()
        )
        return DoctorCheck("hard-link volume compatibility", ok, details)

    def _select_models(
        self, discovered: list[OllamaModel], requested_names: list[str]
    ) -> list[OllamaModel]:
        if not requested_names:
            raise ValueError("provide at least one model name or use --all")

        lookup: dict[str, list[OllamaModel]] = {}
        for model in discovered:
            for key in {
                model.canonical_name,
                model.fully_qualified_name,
                model.short_name,
                model.repository,
            }:
                lookup.setdefault(key, []).append(model)

        selected: list[OllamaModel] = []
        for name in requested_names:
            matches = lookup.get(name, [])
            if not matches:
                available = ", ".join(sorted(lookup.keys()))
                raise ValueError(f"model not found: {name}. Available: {available}")
            selected.extend(matches)

        return selected
