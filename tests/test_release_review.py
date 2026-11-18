"""Regressions for the 0.3.0 pre-release review findings."""

import json
import logging
from dataclasses import replace

import pytest
from conftest import DIGEST_A, DIGEST_B, FakeLMStudio, write_blob, write_manifest

import studiolink.config as config_module
import studiolink.files as files_module
from studiolink.models import LinkMode
from studiolink.service import StudioLinkService
from studiolink.transactions import MARKER, TRANSACTIONS, recover_imports


class FilesystemLMStudio(FakeLMStudio):
    def __init__(self, config):
        super().__init__()
        self.config = config


@pytest.fixture
def library(make_config):
    config = make_config()
    adapter = FilesystemLMStudio(config)
    service = StudioLinkService(config=config, lmstudio=adapter)
    manifest = write_manifest(config)
    blob = write_blob(config, DIGEST_A)
    return config, service, adapter, manifest, blob


def renumber_devices(config):
    """Simulate a reboot that gave the filesystem a new device number."""
    raw = json.loads(config.state_file.read_text())
    for record in raw["sync_records"].values():
        for artifact in record["artifacts"]:
            artifact["imported_identity"][0] += 1
    config.state_file.write_text(json.dumps(raw))


class TestDeviceRenumbering:
    @pytest.mark.parametrize("mode", list(LinkMode))
    def test_imports_stay_synced(self, library, mode):
        config, service, _, _, _ = library
        assert service.sync(sync_all=True, link_mode=mode)[0].status == "synced"
        renumber_devices(config)

        assert service.status()[0].synced is True
        result = service.sync(sync_all=True, link_mode=mode)[0]
        assert result.status == "skipped", result.message

    def test_prune_still_removes_deleted_models(self, library):
        config, service, _, manifest, _ = library
        service.sync(sync_all=True, link_mode=LinkMode.COPY)
        renumber_devices(config)
        manifest.unlink()

        report = service.prune()

        assert report.records_removed == ("llama3:1b",)
        assert not any(config.lmstudio_models_dir.rglob("*.gguf"))

    def test_repull_reclaims_previous_import(self, library):
        config, service, _, _, _ = library
        service.sync(sync_all=True)
        renumber_devices(config)
        write_manifest(config, digest=DIGEST_B)
        write_blob(config, DIGEST_B)

        result = service.sync(sync_all=True)[0]

        assert result.status == "synced", result.message
        assert len(list(config.lmstudio_models_dir.rglob("*.gguf"))) == 1


class TestStoreSelection:
    def test_store_with_models_beats_empty_service_store(self, tmp_path, monkeypatch):
        service_store = tmp_path / "usr-share-ollama" / "models"
        for name in ("blobs", "manifests", "manifests-v2"):
            (service_store / name).mkdir(parents=True)
        home = tmp_path / "home"
        user_store = home / ".ollama" / "models"
        tag = user_store / "manifests" / "registry.ollama.ai" / "library" / "x" / "1b"
        tag.parent.mkdir(parents=True)
        tag.write_text("{}")
        monkeypatch.setattr(config_module, "LINUX_OLLAMA_MODELS_DIRS", (service_store,))
        monkeypatch.setattr(config_module, "_systemd_ollama_models_dir", lambda: None)
        monkeypatch.setattr(config_module.sys, "platform", "linux")

        assert config_module._default_ollama_models_dir(home) == user_store

    def test_empty_store_still_found_when_nothing_has_models(
        self, tmp_path, monkeypatch
    ):
        service_store = tmp_path / "var-lib-ollama"
        (service_store / "blobs").mkdir(parents=True)
        monkeypatch.setattr(config_module, "LINUX_OLLAMA_MODELS_DIRS", (service_store,))
        monkeypatch.setattr(config_module, "_systemd_ollama_models_dir", lambda: None)
        monkeypatch.setattr(config_module.sys, "platform", "linux")

        assert config_module._default_ollama_models_dir(tmp_path / "home") == (
            service_store
        )

    def test_prune_keeps_imports_whose_blobs_live_in_another_store(
        self, library, tmp_path
    ):
        config, service, adapter, _, blob = library
        service.sync(sync_all=True, link_mode=LinkMode.COPY)
        target = next(config.lmstudio_models_dir.rglob("*.gguf"))
        other = tmp_path / "service" / "models"
        (other / "manifests").mkdir(parents=True)
        (other / "blobs").mkdir()
        switched = replace(
            config,
            ollama_models_dir=other,
            ollama_manifests_dir=other / "manifests",
            ollama_manifests_v2_dir=other / "manifests-v2",
            ollama_blobs_dir=other / "blobs",
        )
        adapter.config = switched

        report = StudioLinkService(config=switched, lmstudio=adapter).prune()

        assert target.exists() and blob.exists()
        assert report.records_kept == ("llama3:1b",)
        assert report.records_removed == ()

    def test_prune_removes_once_the_other_store_lost_the_blob(self, library, tmp_path):
        config, service, adapter, manifest, blob = library
        service.sync(sync_all=True, link_mode=LinkMode.COPY)
        manifest.unlink()
        blob.unlink()

        report = service.prune()

        assert report.records_removed == ("llama3:1b",)
        assert report.records_kept == ()


class TestCopyReads:
    @pytest.fixture
    def compares(self, monkeypatch):
        calls = []
        original = files_module.equal_contents

        def counting(left, right):
            calls.append((left, right))
            return original(left, right)

        monkeypatch.setattr(files_module, "equal_contents", counting)
        return calls

    def test_fresh_copy_import_reads_no_file_back(self, library, compares):
        _, service, _, _, _ = library
        assert service.sync(sync_all=True, link_mode=LinkMode.COPY)[0].status == (
            "synced"
        )
        assert compares == []

    def test_resync_uses_saved_identity(self, library, compares):
        _, service, _, _, _ = library
        service.sync(sync_all=True, link_mode=LinkMode.COPY)

        assert service.sync(sync_all=True)[0].status == "skipped"
        assert compares == []

    def test_legacy_copy_is_compared_once_then_remembered(self, library, compares):
        config, service, _, _, _ = library
        service.sync(sync_all=True, link_mode=LinkMode.COPY)
        raw = json.loads(config.state_file.read_text())
        for record in raw["sync_records"].values():
            for artifact in record["artifacts"]:
                artifact["imported_identity"] = None  # as written by 0.2.0
        config.state_file.write_text(json.dumps(raw))

        assert service.sync(sync_all=True)[0].status == "skipped"
        assert len(compares) == 1
        assert service.sync(sync_all=True)[0].status == "skipped"
        assert len(compares) == 1


class TestInterruptedTransactions:
    def leave_crashed_import(self, config):
        work = config.lmstudio_models_dir / TRANSACTIONS / "import-abc"
        work.mkdir(parents=True)
        (work / MARKER).write_text(json.dumps({"filename": "x.gguf"}))
        (work / "x.gguf").write_bytes(b"GGUF")
        (work / "x.gguf.partial").write_bytes(b"junk")
        return work

    def test_recovery_leaves_unknown_files_without_failing(self, library, caplog):
        config, _, _, _, _ = library
        work = self.leave_crashed_import(config)

        with caplog.at_level(logging.WARNING, logger="studiolink"):
            recover_imports(config.lmstudio_models_dir)

        assert sorted(p.name for p in work.iterdir()) == ["x.gguf.partial"]
        assert "import-abc" in caplog.text

    def test_sync_and_prune_continue(self, library):
        config, service, _, _, _ = library
        self.leave_crashed_import(config)

        assert service.sync(sync_all=True)[0].status == "synced"
        service.prune()

    def test_doctor_reports_leftovers(self, library):
        config, service, _, _, _ = library
        self.leave_crashed_import(config)
        recover_imports(config.lmstudio_models_dir)

        check = next(c for c in service.doctor() if c.name == "interrupted imports")

        assert check.ok is False
        assert "import-abc" in check.details

    def test_doctor_quiet_when_clean(self, library):
        _, service, _, _, _ = library
        check = next(c for c in service.doctor() if c.name == "interrupted imports")
        assert check.ok is True
