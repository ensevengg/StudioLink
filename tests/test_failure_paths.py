"""Adversarial checks at filesystem and process boundaries."""

from __future__ import annotations

import errno
import os
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import DIGEST_A, write_blob, write_manifest
from test_import_lifecycle import destination

from studiolink.models import LinkMode
from studiolink.state import StateStore


def test_prune_preserves_untracked_symlink_after_lost_state(library):
    config, service, _, manifest, blob = library
    service.sync(sync_all=True, link_mode=LinkMode.SYMBOLIC_LINK)
    target = destination(config, service.scan()[0])
    config.state_file.unlink()
    manifest.unlink()
    blob.unlink()
    service.prune()
    assert target.is_file(), "prune broke an untracked LM Studio import"


def test_state_write_failure_can_recover_existing_copy(library, monkeypatch):
    config, service, adapter, _, blob = library

    def full(*args, **kwargs):
        raise OSError(errno.ENOSPC, "disk full")

    with monkeypatch.context() as patch:
        patch.setattr(service.state, "_save_raw", full)
        with pytest.raises(OSError, match="disk full"):
            service.sync(sync_all=True, link_mode=LinkMode.COPY)
    target = destination(config, service.scan()[0])
    assert target.read_bytes() == blob.read_bytes()
    assert service.sync(sync_all=True, link_mode=LinkMode.COPY)[0].status == "synced"
    assert len(adapter.calls) == 1


def test_interrupted_atomic_state_write_preserves_old_json(tmp_path, monkeypatch):
    from conftest import make_sync_record

    state = StateStore(tmp_path / "state.json")
    state.upsert(make_sync_record("old:1"))
    old = state.path.read_bytes()

    def failed_replace(*args, **kwargs):
        raise OSError(errno.ENOSPC, "disk full")

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", failed_replace)
        with pytest.raises(OSError):
            state.upsert(make_sync_record("new:1"))
    assert state.path.read_bytes() == old
    assert set(state.get_all_records()) == {"old:1"}
    state.upsert(make_sync_record("new:1"))
    assert set(state.get_all_records()) == {"old:1", "new:1"}


def test_os_releases_lock_after_killed_process(tmp_path):
    state = StateStore(tmp_path / "state.json")
    code = """
import sys, time
from pathlib import Path
from studiolink.state import StateStore
with StateStore(Path(sys.argv[1])).lock():
    print("LOCKED", flush=True)
    time.sleep(30)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    child = subprocess.Popen(
        [sys.executable, "-c", code, str(state.path)],
        env=env,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "LOCKED"
        with pytest.raises(RuntimeError, match="Another StudioLink"):
            with state.lock():
                pytest.fail("overlapping transaction was allowed")
    finally:
        child.kill()
        child.wait(timeout=5)
    with state.lock():
        assert True


@pytest.mark.parametrize("mode", list(LinkMode))
def test_real_cross_filesystem_import(make_config, fake_lmstudio, tmp_path, mode):
    import tempfile

    from studiolink.service import StudioLinkService

    if (
        not Path("/dev/shm").is_dir()
        or Path("/dev/shm").stat().st_dev == tmp_path.stat().st_dev
    ):
        pytest.skip("requires a writable second filesystem")
    with tempfile.TemporaryDirectory(
        prefix="studiolink-test-", dir="/dev/shm"
    ) as second:
        config = make_config(
            lmstudio_models_dir=Path(second) / "models",
            state_dir=Path(second) / "state",
            state_file=Path(second) / "state/state.json",
            import_staging_dir=Path(second) / "state/imports",
        )
        fake_lmstudio.config = config
        service = StudioLinkService(config=config, lmstudio=fake_lmstudio)
        write_manifest(config)
        blob = write_blob(config, DIGEST_A)
        assert blob.stat().st_dev != config.state_dir.parent.stat().st_dev
        result = service.sync(sync_all=True, link_mode=mode)[0]
        if mode is LinkMode.HARD_LINK:
            assert result.status == "error"
            assert blob.stat().st_nlink == 1
        else:
            assert result.status == "synced", result.message
            assert (
                destination(config, service.scan()[0]).read_bytes() == blob.read_bytes()
            )


def test_failed_prune_can_be_retried_without_losing_record(library, monkeypatch):
    config, service, _, manifest, blob = library
    service.sync(sync_all=True)
    target = destination(config, service.scan()[0])
    alias = config.import_staging_dir / service.scan()[0].import_filename
    manifest.unlink()
    blob.unlink()
    original = Path.unlink

    def denied(path, *args, **kwargs):
        if path == target:
            raise PermissionError("busy destination")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", denied)
        report = service.prune()
        assert not report.imports[0].removed
    assert service.state.get_all_records()
    assert alias.is_file() and target.is_file()
    service.prune()
    assert not service.state.get_all_records()
    assert not target.exists() and not alias.exists()


def test_unavailable_lmstudio_directory_preserves_aliases_and_state(library):
    config, service, _, manifest, blob = library
    service.sync(sync_all=True)
    alias = config.import_staging_dir / service.scan()[0].import_filename
    manifest.unlink()
    blob.unlink()
    config.lmstudio_models_dir.rename(config.lmstudio_models_dir.with_name("unmounted"))
    before = config.state_file.read_bytes()
    with pytest.raises(RuntimeError, match="LM Studio models directory is unavailable"):
        service.prune()
    assert alias.is_file()
    assert config.state_file.read_bytes() == before


def test_repull_preserves_untracked_symbolic_dependency(library):
    from conftest import DIGEST_B

    config, service, _, _, _ = library
    service.sync(sync_all=True, link_mode=LinkMode.SYMBOLIC_LINK)
    alias = config.import_staging_dir / service.scan()[0].import_filename
    manual = config.lmstudio_models_dir / "manual" / "model" / "saved.gguf"
    manual.parent.mkdir(parents=True)
    manual.symlink_to(alias)
    original = manual.read_bytes()
    write_manifest(config, digest=DIGEST_B)
    write_blob(config, DIGEST_B)
    assert (
        service.sync(sync_all=True, link_mode=LinkMode.SYMBOLIC_LINK)[0].status
        == "synced"
    )
    assert manual.read_bytes() == original


def test_prune_cleans_abandoned_copy_staging(library):
    config, service, _, _, blob = library
    model = service.scan()[0]
    staging = config.ollama_models_dir / ".studiolink-imports"
    staging.mkdir()
    alias = staging / model.import_filename
    os.link(blob, alias)
    config.lmstudio_models_dir.mkdir(parents=True)
    report = service.prune(dry_run=True)
    assert alias.exists()
    assert alias in {item.path for item in report.aliases}
    service.prune()
    assert not alias.exists()
    assert blob.stat().st_nlink == 1
