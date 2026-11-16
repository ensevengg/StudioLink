"""Repeat mixed user actions to catch interactions between lifecycle operations."""

import hashlib
import random

import pytest
from conftest import write_blob, write_manifest

from studiolink.models import LinkMode


@pytest.mark.parametrize("seed", range(10))
def test_mixed_operations_remain_repairable_and_reclaimable(library, seed):
    config, service, _, original_manifest, _ = library
    original_manifest.unlink()
    rng = random.Random(seed)
    versions = {}
    for tag in ("small", "latest"):
        content = b"GGUF" + bytes([1]) * 8
        digest = "sha256:" + hashlib.sha256(content).hexdigest()
        write_blob(config, digest).write_bytes(content)
        write_manifest(config, tag=tag, digest=digest)
        versions[tag] = 1
    for step in range(30):
        action = rng.choice(
            ("repull", "remove-target", "state-loss", "dry-run", "sync")
        )
        if action == "repull":
            tag = rng.choice(list(versions))
            versions[tag] += 1
            content = b"GGUF" + bytes([versions[tag]]) * 8
            digest = "sha256:" + hashlib.sha256(content).hexdigest()
            # Ollama reuses an existing content-addressed blob. Do not rewrite
            # its inode (and every hard-linked import) when another tag uses it.
            blob = config.ollama_blobs_dir / digest.replace(":", "-")
            if not blob.exists():
                write_blob(config, digest).write_bytes(content)
            write_manifest(config, tag=tag, digest=digest)
        elif action == "remove-target":
            targets = list(config.lmstudio_models_dir.glob("*/*/*.gguf"))
            if targets:
                rng.choice(targets).unlink()
        elif action == "state-loss":
            # Losing state after an update leaves obsolete imports untracked:
            # clean them while ownership is still available before this step.
            service.prune() if config.lmstudio_models_dir.exists() else None
            config.state_file.unlink(missing_ok=True)
        mode = rng.choice(list(LinkMode))
        if action == "dry-run":
            before = {
                p: p.lstat().st_ino
                for p in config.lmstudio_models_dir.glob("*/*/*.gguf")
            }
            service.sync(sync_all=True, link_mode=mode, dry_run=True)
            after = {
                p: p.lstat().st_ino
                for p in config.lmstudio_models_dir.glob("*/*/*.gguf")
            }
            assert before == after
        results = service.sync(sync_all=True, link_mode=mode)
        assert all(item.status in ("synced", "skipped") for item in results), (
            step,
            action,
            mode,
            [(item.status, item.message) for item in results],
        )
        assert all(entry.synced for entry in service.status())
    # All library models removed: every tracked destination must disappear.
    for model in service.scan():
        model.manifest_path.unlink()
    service.prune()
    assert not service.state.get_all_records()
    assert not list(config.lmstudio_models_dir.glob("*/*/*.gguf"))
    assert not list(config.import_staging_dir.glob("*.gguf"))
