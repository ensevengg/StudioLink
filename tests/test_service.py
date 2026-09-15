import os
import types
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import (
    DIGEST_A,
    make_sync_record,
    write_blob,
    write_manifest,
)

import studiolink.service as service_module
from studiolink.models import LinkMode
from studiolink.ollama_adapter import OllamaAdapter
from studiolink.service import StudioLinkService
from studiolink.state import StateStore


def build_service(config, lmstudio):
    return StudioLinkService(
        config=config, ollama=OllamaAdapter(config), lmstudio=lmstudio
    )


class TestDoctorVolumeCheck:
    def test_same_device_passes(self, make_config, fake_lmstudio):
        config = make_config()
        for directory in (
            config.ollama_blobs_dir,
            config.import_staging_dir,
            config.lmstudio_models_dir,
        ):
            directory.mkdir(parents=True)
        service = build_service(config, fake_lmstudio)

        check = service._volume_compatibility_check()

        assert check.ok is True
        assert "device" in check.details

    def test_mismatched_devices_fail(self, make_config, fake_lmstudio, monkeypatch):
        config = make_config()
        for directory in (
            config.ollama_blobs_dir,
            config.import_staging_dir,
            config.lmstudio_models_dir,
        ):
            directory.mkdir(parents=True)
        service = build_service(config, fake_lmstudio)

        real_stat = os.stat
        devices = {}

        def fake_stat(path, *args, **kwargs):
            real_stat(path, *args, **kwargs)
            key = str(path)
            if key not in devices:
                devices[key] = len(devices) + 1
            return types.SimpleNamespace(st_dev=devices[key])

        monkeypatch.setattr(service_module.os, "stat", fake_stat)
        check = service._volume_compatibility_check()

        assert check.ok is False

    def test_skips_when_directories_missing(self, make_config, fake_lmstudio):
        service = build_service(make_config(), fake_lmstudio)
        check = service._volume_compatibility_check()
        assert check.ok is True
        assert "skipped" in check.details

    def test_doctor_accepts_import_directory_that_has_not_been_created_yet(
        self, make_config, fake_lmstudio
    ):
        config = make_config()
        config.ollama_manifests_dir.mkdir(parents=True)
        config.ollama_blobs_dir.mkdir(parents=True)
        config.lmstudio_models_dir.mkdir(parents=True)
        service = build_service(config, fake_lmstudio)

        checks = service.doctor()

        aliases = next(c for c in checks if c.name == "import alias directory")
        assert aliases.ok is True
        assert "will be created" in aliases.details

    def test_doctor_includes_volume_check(self, make_config, fake_lmstudio):
        config = make_config()
        for directory in (
            config.ollama_blobs_dir,
            config.import_staging_dir,
            config.lmstudio_models_dir,
        ):
            directory.mkdir(parents=True)
        service = build_service(config, fake_lmstudio)

        checks = service.doctor()
        volume = next(c for c in checks if c.name == "hard-link volume compatibility")
        assert volume.ok is True


class TestPrune:
    def test_removes_orphaned_alias(self, make_config, fake_lmstudio):
        config = make_config()
        service = build_service(config, fake_lmstudio)
        write_manifest(config, digest=DIGEST_A)
        write_blob(config, DIGEST_A)

        config.import_staging_dir.mkdir(parents=True)
        orphan = config.import_staging_dir / "oldmodel-8b-000000000000.gguf"
        orphan.write_bytes(b"GGUF" + b"x" * 100)

        report = service.prune()

        assert not orphan.exists()
        assert len(report.aliases) == 1
        assert report.aliases[0].removed is True
        assert report.aliases[0].would_free is True  # last link
        assert report.aliases[0].size == 104

    def test_refuses_prune_after_partial_manifest_scan(
        self, make_config, fake_lmstudio, monkeypatch
    ):
        config = make_config()
        service = build_service(config, fake_lmstudio)
        broken = write_manifest(config, repository="broken", digest=DIGEST_A)
        write_manifest(config, repository="healthy", digest=DIGEST_A)
        write_blob(config, DIGEST_A)
        config.import_staging_dir.mkdir(parents=True)
        alias = config.import_staging_dir / "orphan.gguf"
        alias.write_bytes(b"GGUFxxxx")
        real_read_text = Path.read_text

        def flaky_read_text(path, *args, **kwargs):
            if path == broken:
                raise OSError("access denied")
            return real_read_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", flaky_read_text)

        with pytest.raises(RuntimeError, match="scan was incomplete"):
            service.prune()

        assert alias.exists()

    def test_preserves_alias_when_current_manifest_is_not_ready(
        self, make_config, fake_lmstudio
    ):
        config = make_config()
        service = build_service(config, fake_lmstudio)
        write_manifest(config, digest=DIGEST_A)
        blob = write_blob(config, DIGEST_A)
        model = OllamaAdapter(config).scan_models()[0]
        config.import_staging_dir.mkdir(parents=True)
        alias = config.import_staging_dir / model.import_filename
        os.link(blob, alias)
        StateStore(config.state_file).upsert(
            replace(
                make_sync_record(model.canonical_name, DIGEST_A),
                import_alias_path=alias,
            )
        )
        blob.unlink()

        report = service.prune()

        assert report.aliases == ()
        assert alias.exists()

    def test_preserves_alias_required_by_symbolic_import(
        self, make_config, fake_lmstudio
    ):
        config = make_config()
        service = build_service(config, fake_lmstudio)
        manifest = write_manifest(config, digest=DIGEST_A)
        blob = write_blob(config, DIGEST_A)
        model = OllamaAdapter(config).scan_models()[0]

        config.import_staging_dir.mkdir(parents=True)
        alias = config.import_staging_dir / model.import_filename
        os.link(blob, alias)
        record = replace(
            make_sync_record(model.canonical_name, DIGEST_A),
            import_alias_path=alias,
            link_mode=LinkMode.SYMBOLIC_LINK,
        )
        StateStore(config.state_file).upsert(record)
        manifest.unlink()
        blob.unlink()

        report = service.prune()

        assert report.aliases == ()
        assert alias.exists()

    def test_keeps_healthy_alias(self, make_config, fake_lmstudio):
        config = make_config()
        service = build_service(config, fake_lmstudio)
        write_manifest(config, digest=DIGEST_A)
        blob = write_blob(config, DIGEST_A)
        model = OllamaAdapter(config).scan_models()[0]

        config.import_staging_dir.mkdir(parents=True)
        os.link(blob, config.import_staging_dir / model.import_filename)

        report = service.prune()

        assert report.aliases == ()
        assert (config.import_staging_dir / model.import_filename).exists()

    def test_removes_same_name_alias_pointing_elsewhere(
        self, make_config, fake_lmstudio
    ):
        config = make_config()
        service = build_service(config, fake_lmstudio)
        write_manifest(config, digest=DIGEST_A)
        write_blob(config, DIGEST_A)
        model = OllamaAdapter(config).scan_models()[0]

        config.import_staging_dir.mkdir(parents=True)
        stale = config.import_staging_dir / model.import_filename
        stale.write_bytes(b"GGUF-but-not-the-current-blob")

        report = service.prune()

        assert len(report.aliases) == 1
        assert "no longer points" in report.aliases[0].reason
        assert not stale.exists()

    def test_dry_run_removes_nothing(self, make_config, fake_lmstudio):
        config = make_config()
        service = build_service(config, fake_lmstudio)
        write_manifest(config, digest=DIGEST_A)
        write_blob(config, DIGEST_A)

        config.import_staging_dir.mkdir(parents=True)
        orphan = config.import_staging_dir / "gone-8b-000000000000.gguf"
        orphan.write_bytes(b"GGUFxxxx")

        report = service.prune(dry_run=True)

        assert orphan.exists()
        assert report.dry_run is True
        assert report.aliases[0].removed is False

    def test_removes_stale_records_keeps_current(self, make_config, fake_lmstudio):
        config = make_config()
        service = build_service(config, fake_lmstudio)
        write_manifest(config, repository="llama3", tag="1b", digest=DIGEST_A)
        write_blob(config, DIGEST_A)

        store = StateStore(config.state_file)
        store.upsert(make_sync_record("llama3:1b", DIGEST_A))
        store.upsert(make_sync_record("removed:model", DIGEST_A))

        report = service.prune()

        assert report.records_removed == ("removed:model",)
        records = store.get_all_records()
        assert set(records) == {"llama3:1b"}

    def test_refuses_prune_when_ollama_library_is_unavailable(
        self, make_config, fake_lmstudio
    ):
        # A missing models directory is not proof that the user removed every
        # model. Prune must not delete either aliases or state in that case.
        config = make_config()
        service = build_service(config, fake_lmstudio)
        store = StateStore(config.state_file)
        store.upsert(make_sync_record("llama3:1b", DIGEST_A))
        config.import_staging_dir.mkdir(parents=True)
        alias = config.import_staging_dir / "llama3-1b-old.gguf"
        alias.write_bytes(b"GGUFxxxx")

        with pytest.raises(RuntimeError, match="Ollama library could not be scanned"):
            service.prune()

        assert alias.exists()
        assert set(store.get_all_records()) == {"llama3:1b"}

    def test_empty_but_available_library_removes_non_symbolic_records(
        self, make_config, fake_lmstudio
    ):
        config = make_config()
        config.ollama_manifests_dir.mkdir(parents=True)
        service = build_service(config, fake_lmstudio)
        store = StateStore(config.state_file)
        store.upsert(make_sync_record("removed:model", DIGEST_A))

        report = service.prune()

        assert report.records_removed == ("removed:model",)
        assert store.get_all_records() == {}

    def test_reports_space_freed_for_last_link(self, make_config, fake_lmstudio):
        # Simulate `ollama rm`: manifest and blob deleted, but the hard-linked
        # alias keeps the bytes on disk. Prune reclaims them.
        config = make_config()
        service = build_service(config, fake_lmstudio)
        write_manifest(config, repository="llama3", tag="1b", digest=DIGEST_A)
        blob = write_blob(config, DIGEST_A)
        model = OllamaAdapter(config).scan_models()[0]

        config.import_staging_dir.mkdir(parents=True)
        alias = config.import_staging_dir / model.import_filename
        os.link(blob, alias)
        blob.unlink()
        model.manifest_path.unlink()

        report = service.prune()

        assert not alias.exists()
        assert report.aliases[0].would_free is True
        assert report.aliases[0].size == 12


class TestStatusAndSelection:
    def test_status_reports_pending(self, make_config, fake_lmstudio):
        config = make_config()
        service = build_service(config, fake_lmstudio)
        write_manifest(config, digest=DIGEST_A)
        write_blob(config, DIGEST_A)

        entries = service.status()

        assert len(entries) == 1
        assert entries[0].display_status == "pending"
        assert entries[0].synced is False

    def test_status_reports_synced_only_when_imported_target_exists(
        self, make_config, fake_lmstudio
    ):
        config = make_config()
        service = build_service(config, fake_lmstudio)
        write_manifest(config, digest=DIGEST_A)
        write_blob(config, DIGEST_A)
        model = OllamaAdapter(config).scan_models()[0]
        target = config.lmstudio_models_dir / model.user_repo / model.import_filename
        target.parent.mkdir(parents=True)
        target.write_bytes(b"GGUFxxxx")
        StateStore(config.state_file).upsert(
            replace(
                make_sync_record("llama3:1b", DIGEST_A),
                imported_model_path=target,
            )
        )

        entries = service.status()

        assert entries[0].display_status == "synced"
        assert entries[0].synced is True

    def test_status_reconciles_legacy_record_with_derived_target(
        self, make_config, fake_lmstudio
    ):
        config = make_config()
        service = build_service(config, fake_lmstudio)
        write_manifest(config, digest=DIGEST_A)
        write_blob(config, DIGEST_A)
        record = make_sync_record("llama3:1b", DIGEST_A)
        target = (
            config.lmstudio_models_dir
            / record.user_repo
            / record.import_alias_path.name
        )
        target.parent.mkdir(parents=True)
        target.write_bytes(b"GGUFxxxx")
        StateStore(config.state_file).upsert(record)

        entries = service.status()

        assert entries[0].display_status == "synced"
        assert entries[0].synced is True

        target.unlink()
        entries = service.status()

        assert entries[0].display_status == "pending"
        assert entries[0].synced is False

    def test_status_reports_pending_when_imported_target_was_deleted(
        self, make_config, fake_lmstudio
    ):
        config = make_config()
        service = build_service(config, fake_lmstudio)
        write_manifest(config, digest=DIGEST_A)
        write_blob(config, DIGEST_A)
        missing_target = config.lmstudio_models_dir / "ollama" / "llama3" / "gone.gguf"
        StateStore(config.state_file).upsert(
            replace(
                make_sync_record("llama3:1b", DIGEST_A),
                imported_model_path=missing_target,
            )
        )

        entries = service.status()

        assert entries[0].display_status == "pending"
        assert entries[0].synced is False

    def test_select_models_rejects_unknown_name(self, make_config, fake_lmstudio):
        service = build_service(make_config(), fake_lmstudio)
        try:
            service._select_models([], ["nope"])
        except ValueError as exc:
            assert "model not found" in str(exc)
        else:
            raise AssertionError("expected ValueError")

    def test_select_models_matches_repository_all_tags(
        self, make_config, fake_lmstudio
    ):
        config = make_config()
        service = build_service(config, fake_lmstudio)
        write_manifest(config, repository="llama3", tag="1b", digest=DIGEST_A)
        write_manifest(config, repository="llama3", tag="70b", digest=DIGEST_A)
        models = OllamaAdapter(config).scan_models()

        selected = service._select_models(models, ["llama3"])

        assert {m.tag for m in selected} == {"1b", "70b"}
