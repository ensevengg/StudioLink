"""Regressions for the 0.3.0 pre-release review findings."""

import json
import logging
import os
from dataclasses import replace
from pathlib import Path

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


class TestCorruptStateFile:
    def corrupt(self, config, text="{ not json"):
        config.state_file.parent.mkdir(parents=True, exist_ok=True)
        config.lmstudio_models_dir.mkdir(parents=True, exist_ok=True)
        config.state_file.write_text(text)

    def backups(self, config):
        return list(config.state_file.parent.glob("state.json.corrupt-*"))

    def test_unreadable_state_is_kept_aside(self, library):
        config, service, _, _, _ = library
        self.corrupt(config)

        assert service.sync(sync_all=True)[0].status == "synced"

        [backup] = self.backups(config)
        assert backup.read_text() == "{ not json"
        assert "llama3:1b" in json.loads(config.state_file.read_text())["sync_records"]

    def test_non_object_records_are_kept_aside_too(self, library):
        config, service, _, _, _ = library
        self.corrupt(config, '{"sync_records": ["legacy-junk"]}')

        assert service.sync(sync_all=True)[0].status == "synced"

        [backup] = self.backups(config)
        assert "legacy-junk" in backup.read_text()

    def test_read_only_commands_leave_it_in_place(self, library):
        # An unlocked reader racing a sync must never move the file aside.
        config, service, _, _, _ = library
        self.corrupt(config)

        service.status()
        service.sync(sync_all=True, dry_run=True)
        service.prune(dry_run=True)

        assert self.backups(config) == []
        assert config.state_file.read_text() == "{ not json"

    def test_refuses_before_importing_when_it_cannot_back_up(
        self, library, monkeypatch
    ):
        config, service, _, _, _ = library
        self.corrupt(config, "[]")
        real_replace = os.replace

        def blocked(src, dst):
            if ".corrupt-" in str(dst):
                raise PermissionError("read-only")
            return real_replace(src, dst)

        monkeypatch.setattr(os, "replace", blocked)

        with pytest.raises(RuntimeError, match="unreadable state file"):
            service.sync(sync_all=True)
        with pytest.raises(RuntimeError, match="unreadable state file"):
            service.prune()
        assert config.state_file.read_text() == "[]"
        assert not any(config.lmstudio_models_dir.rglob("*.gguf"))

    def test_file_vanishing_mid_read_is_empty_state(self, library, monkeypatch):
        config, service, _, _, _ = library
        service.sync(sync_all=True)  # the file exists until the read

        def gone(self, *args, **kwargs):
            raise FileNotFoundError(self)

        monkeypatch.setattr(type(config.state_file), "read_text", gone)
        assert service.state.get_all_records() == {}


class TestReclaimKeepsUserSymlinkTargets:
    def repull(self, config):
        write_manifest(config, digest=DIGEST_B)
        write_blob(config, DIGEST_B)

    @pytest.mark.parametrize("mode", list(LinkMode))
    def test_repull_keeps_import_a_user_symlink_needs(self, library, mode):
        config, service, _, _, _ = library
        service.sync(sync_all=True, link_mode=mode)
        old = next(config.lmstudio_models_dir.rglob("*.gguf"))
        mine = config.lmstudio_models_dir / "mine" / "favourite.link"
        mine.parent.mkdir()
        mine.symlink_to(old)
        self.repull(config)

        result = service.sync(sync_all=True, link_mode=mode)[0]

        assert result.status == "synced", result.message
        assert old.exists() and mine.resolve() == old.resolve()
        assert len(list(config.lmstudio_models_dir.rglob("*.gguf"))) == 2

    def test_link_spelled_through_the_real_path(self, make_config, tmp_path):
        real = tmp_path / "disk" / "models"
        real.mkdir(parents=True)
        linked = tmp_path / "lmstudio" / "models"
        linked.parent.mkdir()
        linked.symlink_to(real, target_is_directory=True)
        config = make_config(lmstudio_models_dir=linked)
        service = StudioLinkService(config=config, lmstudio=FilesystemLMStudio(config))
        write_manifest(config)
        write_blob(config, DIGEST_A)
        service.sync(sync_all=True, link_mode=LinkMode.COPY)
        old = next(real.rglob("*.gguf"))
        mine = real / "mine.link"
        mine.symlink_to(old.resolve())
        self.repull(config)

        assert service.sync(sync_all=True)[0].status == "synced"
        assert old.exists()

    def test_directory_link_protects_files_inside(self, library):
        config, service, _, _, _ = library
        service.sync(sync_all=True, link_mode=LinkMode.COPY)
        old = next(config.lmstudio_models_dir.rglob("*.gguf"))
        mine = config.lmstudio_models_dir / "mine"
        mine.symlink_to(old.parent, target_is_directory=True)
        self.repull(config)

        assert service.sync(sync_all=True)[0].status == "synced"
        assert old.exists()

    def test_unscannable_folder_keeps_old_import_and_sync_succeeds(
        self, library, monkeypatch
    ):
        config, service, _, _, _ = library
        service.sync(sync_all=True, link_mode=LinkMode.COPY)
        old = next(config.lmstudio_models_dir.rglob("*.gguf"))
        real_walk = files_module.os.walk

        def walk(top, onerror=None, **kwargs):
            if Path(top) == config.lmstudio_models_dir and onerror is not None:
                onerror(PermissionError(13, "Permission denied", f"{top}/locked"))
            yield from real_walk(top, onerror=onerror, **kwargs)

        monkeypatch.setattr(files_module.os, "walk", walk)
        self.repull(config)

        assert service.sync(sync_all=True)[0].status == "synced"
        assert old.exists()
        with pytest.raises(RuntimeError, match="Could not scan"):
            service.prune()


class TestPruneExplainsKeptFiles:
    @pytest.fixture(params=["real-root", "symlinked-root"])
    def library(self, request, make_config, tmp_path):
        overrides = {}
        if request.param == "symlinked-root":
            real = tmp_path / "disk" / "models"
            real.mkdir(parents=True)
            linked = tmp_path / "lmstudio" / "models"
            linked.parent.mkdir()
            linked.symlink_to(real, target_is_directory=True)
            overrides["lmstudio_models_dir"] = linked
        config = make_config(**overrides)
        service = StudioLinkService(config=config, lmstudio=FilesystemLMStudio(config))
        manifest = write_manifest(config)
        blob = write_blob(config, DIGEST_A)
        return config, service, None, manifest, blob

    def test_link_to_studiolink_folder_keeps_files_and_says_why(self, library, caplog):
        config, service, _, manifest, _ = library
        service.sync(sync_all=True, link_mode=LinkMode.COPY)
        old = next(config.lmstudio_models_dir.rglob("*.gguf"))
        (config.lmstudio_models_dir / "ollama-link").symlink_to(
            config.lmstudio_models_dir / "ollama", target_is_directory=True
        )
        manifest.unlink()

        with caplog.at_level(logging.WARNING, logger="studiolink"):
            report = service.prune()

        assert old.exists()
        assert report.records_removed == ()
        assert f"Keeping {old}" in caplog.text

    def test_kept_alias_is_reported_once(self, library, caplog):
        config, service, _, manifest, _ = library
        service.sync(sync_all=True, link_mode=LinkMode.HARD_LINK)
        [alias] = config.import_staging_dir.iterdir()
        (config.lmstudio_models_dir / "mine.link").symlink_to(alias)
        manifest.unlink()

        with caplog.at_level(logging.WARNING, logger="studiolink"):
            service.prune()

        assert alias.exists()
        assert caplog.text.count(f"Keeping {alias}") == 1
