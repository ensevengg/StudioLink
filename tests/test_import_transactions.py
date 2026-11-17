"""Failures in the actual adapter must never leave a permanent partial model."""

from __future__ import annotations

import errno
import shutil
import subprocess
from pathlib import Path

import pytest
from conftest import DIGEST_A, write_blob, write_manifest

import studiolink.lmstudio_adapter as adapter_module
from studiolink.models import LinkMode
from studiolink.service import StudioLinkService


@pytest.mark.parametrize("failure", ["disk-full", "timeout", "interrupt"])
def test_partial_cli_copy_is_removed_and_retry_works(make_config, monkeypatch, failure):
    config = make_config()
    monkeypatch.setattr(
        adapter_module,
        "lmstudio_models_directory",
        lambda home: config.lmstudio_models_dir,
    )
    service = StudioLinkService(config=config)
    write_manifest(config)
    blob = write_blob(config, DIGEST_A)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        source = Path(command[2])
        target = (
            config.lmstudio_models_dir
            / command[command.index("--user-repo") + 1]
            / source.name
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        if len(calls) == 1:
            target.write_bytes(source.read_bytes()[:5])
            if failure == "timeout":
                raise subprocess.TimeoutExpired(command, 3600)
            if failure == "interrupt":
                raise KeyboardInterrupt
            raise subprocess.CalledProcessError(1, command, stderr="disk full")
        shutil.copyfile(source, target)
        return subprocess.CompletedProcess(command, 0, stdout="copied", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    if failure == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            service.sync(sync_all=True, link_mode=LinkMode.COPY)
    else:
        assert service.sync(sync_all=True, link_mode=LinkMode.COPY)[0].status == "error"
    assert not list(config.lmstudio_models_dir.rglob("*.gguf"))
    assert blob.stat().st_nlink == 1
    assert service.sync(sync_all=True, link_mode=LinkMode.COPY)[0].status == "synced"
    assert len(list(config.lmstudio_models_dir.rglob("*.gguf"))) == 1


def test_killed_process_leaves_recoverable_temporary_import(tmp_path):
    import os
    import sys

    from studiolink.transactions import recover_imports

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    code = """
import os, sys
from pathlib import Path
from studiolink.transactions import import_transaction
with import_transaction(Path(sys.argv[1]), 'weights.gguf') as target:
    target.write_bytes(b'GGUFpartial')
    os._exit(19)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)], env=env, timeout=10
    )
    assert result.returncode == 19
    assert len(list(tmp_path.rglob("*.gguf"))) == 1
    recover_imports(tmp_path)
    assert not list(tmp_path.rglob("*.gguf"))
    assert not list((tmp_path / ".studiolink-transactions").iterdir())


def test_recovery_preserves_active_transaction(tmp_path):
    from studiolink.transactions import import_transaction, recover_imports

    with import_transaction(tmp_path, "weights.gguf") as target:
        target.write_bytes(b"GGUFpartial")
        recover_imports(tmp_path)
        assert target.is_file()
    assert not target.exists()


def test_publication_never_overwrites_existing_destination(tmp_path):
    from studiolink.transactions import publish_import

    staged = tmp_path / "staged"
    staged.write_bytes(b"GGUFnew")
    target = tmp_path / "existing"
    target.write_bytes(b"user data")
    with pytest.raises(FileExistsError):
        publish_import(staged, target)
    assert target.read_bytes() == b"user data"
    assert staged.read_bytes() == b"GGUFnew"


def test_copy_publication_works_without_filesystem_hardlinks(tmp_path, monkeypatch):
    import os

    from studiolink.transactions import publish_import

    staged = tmp_path / "staged"
    staged.write_bytes(b"GGUFnew")
    target = tmp_path / "model"

    def unsupported(*args, **kwargs):
        raise OSError(errno.EOPNOTSUPP, "hardlinks unsupported")

    monkeypatch.setattr(os, "link", unsupported)
    publish_import(staged, target)
    assert target.read_bytes() == b"GGUFnew"
    assert not staged.exists()


def test_prune_reports_and_removes_abandoned_partial_import(library):
    import json

    from studiolink.transactions import TRANSACTIONS

    config, service, _, _, _ = library
    work = config.lmstudio_models_dir / TRANSACTIONS / "import-interrupted"
    work.mkdir(parents=True)
    target = work / "model.gguf"
    target.write_bytes(b"GGUFpartial")
    (work / "owner.json").write_text(json.dumps({"filename": target.name}))
    report = service.prune(dry_run=True)
    assert target.exists()
    assert len(report.imports) == 1
    assert report.imports[0].would_free and not report.imports[0].removed
    report = service.prune()
    assert not target.exists()
    assert report.imports[0].removed and report.imports[0].size == 11
