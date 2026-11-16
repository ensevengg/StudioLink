"""Regression tests for PR #5 review findings."""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path

import pytest
from conftest import (
    DIGEST_A,
    DIGEST_B,
    make_sync_record,
    write_blob,
    write_manifest,
)
from test_import_lifecycle import destination

import studiolink.syncer as syncer_module
from studiolink.files import artifact_destination, destination_path
from studiolink.models import LinkMode, SyncedArtifact
from studiolink.state import StateStore
from studiolink.transactions import TRANSACTIONS

PROJECTOR_DIGEST = "sha256:" + "c" * 64


def write_bundle(config, model_digest):
    write_manifest(
        config,
        repository="llava",
        content={
            "layers": [
                {
                    "mediaType": "application/vnd.ollama.image.model",
                    "digest": model_digest,
                },
                {
                    "mediaType": "application/vnd.ollama.image.projector",
                    "digest": PROJECTOR_DIGEST,
                },
            ]
        },
    )
    write_blob(config, model_digest)
    write_blob(config, PROJECTOR_DIGEST)


def symlink_or_skip(link: Path, target: Path, **kwargs) -> None:
    try:
        link.symlink_to(target, **kwargs)
    except (OSError, NotImplementedError) as exc:  # pragma: no cover - Windows
        pytest.skip(f"symlinks unavailable: {exc}")


def test_sync_rejects_repository_directory_escaping_models_root(library, tmp_path):
    config, service, adapter, _, _ = library
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    config.lmstudio_models_dir.mkdir(parents=True)
    symlink_or_skip(
        config.lmstudio_models_dir / "ollama", outside, target_is_directory=True
    )

    result = service.sync(sync_all=True)[0]

    assert result.status == "error"
    assert "outside the configured models directory" in result.message
    assert adapter.calls == []
    assert not list(outside.iterdir())
    assert StateStore(config.state_file).get_all_records() == {}


def test_sync_dry_run_reports_escaping_destination(library, tmp_path):
    config, service, _, _, _ = library
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    config.lmstudio_models_dir.mkdir(parents=True)
    symlink_or_skip(
        config.lmstudio_models_dir / "ollama", outside, target_is_directory=True
    )

    result = service.sync(sync_all=True, dry_run=True)[0]

    assert result.status == "error"


def test_destinations_share_one_spelling_through_a_symlinked_root(tmp_path):
    real = tmp_path / "real-models"
    real.mkdir()
    alias_root = tmp_path / "models"
    symlink_or_skip(alias_root, real, target_is_directory=True)
    recorded = real / "ollama" / "llama3" / "weights.gguf"
    record = make_sync_record()
    artifact = SyncedArtifact(
        role=record.primary_artifact.role,
        digest=DIGEST_A,
        blob_path=Path("blob"),
        import_alias_path=Path("weights.gguf"),
        imported_model_path=recorded,
    )

    spelled = artifact_destination(record, artifact, alias_root)

    assert spelled == alias_root.absolute() / "ollama" / "llama3" / "weights.gguf"
    assert spelled == destination_path(alias_root, "ollama/llama3", "weights.gguf")


def test_registry_port_never_reaches_repository_name():
    from conftest import make_ollama_model

    model = make_ollama_model(
        "localhost:5000/team/coder:1b",
        registry="localhost:5000",
        namespace="team",
        repository="coder",
        tag="1b",
    )

    assert ":" not in model.user_repo
    assert model.user_repo == "ollama/localhost-5000-team-coder"


def test_dry_run_counts_inode_shared_with_recovered_transaction_once(library):
    config, service, _, _, _ = library
    work = config.lmstudio_models_dir / TRANSACTIONS / "import-interrupted"
    work.mkdir(parents=True)
    partial = work / "model.gguf"
    partial.write_bytes(b"GGUFpartial")
    (work / "owner.json").write_text(json.dumps({"filename": partial.name}))
    config.import_staging_dir.mkdir(parents=True)
    orphan = config.import_staging_dir / "gone-1b-000000000000.gguf"
    os.link(partial, orphan)

    report = service.prune(dry_run=True)

    entries = [*report.imports, *report.aliases]
    assert {entry.path for entry in entries} == {partial, orphan}
    assert sum(entry.would_free for entry in entries) == 1


def test_prune_keeps_destination_an_untracked_symlink_depends_on(library):
    config, service, _, manifest, blob = library
    service.sync(sync_all=True)
    target = destination(config, service.scan()[0])
    manual = config.lmstudio_models_dir / "manual" / "model" / "saved.gguf"
    manual.parent.mkdir(parents=True)
    symlink_or_skip(manual, target)
    manifest.unlink()
    blob.unlink()

    report = service.prune()

    assert target.is_file()
    assert manual.read_bytes() == b"GGUF" + b"0" * 8
    assert report.records_removed == ()
    assert set(StateStore(config.state_file).get_all_records()) == {"llama3:1b"}


def test_rejected_replacement_bundle_keeps_previous_bundle(make_syncer, monkeypatch):
    from test_syncer import scan_models, sync_kwargs

    syncer, config, fake = make_syncer()
    write_bundle(config, DIGEST_A)
    first = syncer.sync(scan_models(config), **sync_kwargs())[0]
    assert first.status == "synced"
    store = StateStore(config.state_file)
    previous = store.get_record("llava:1b")
    old_targets = [
        artifact_destination(previous, artifact, config.lmstudio_models_dir)
        for artifact in previous.artifacts
    ]
    write_bundle(config, DIGEST_B)
    fake.inventory_override = ()
    monkeypatch.setattr(syncer_module.time, "sleep", lambda _: None)

    result = syncer.sync(scan_models(config), **sync_kwargs())[0]

    assert result.status == "error"
    assert "vision-capable" in result.message
    assert store.get_record("llava:1b") == previous
    assert all(target.is_file() for target in old_targets)


def test_copy_reuse_falls_back_when_hardlinks_are_unsupported(make_syncer, monkeypatch):
    from test_syncer import scan_models, sync_kwargs

    syncer, config, fake = make_syncer()
    write_manifest(config, tag="1b", digest=DIGEST_A)
    write_manifest(config, tag="latest", digest=DIGEST_A)
    write_blob(config, DIGEST_A)
    real_link = os.link

    def no_hardlinks_in_lmstudio(source, target, *args, **kwargs):
        if Path(target).is_relative_to(config.lmstudio_models_dir):
            raise OSError(errno.EPERM, "hard links not permitted")
        return real_link(source, target, *args, **kwargs)

    monkeypatch.setattr(syncer_module.os, "link", no_hardlinks_in_lmstudio)

    results = syncer.sync(scan_models(config), **sync_kwargs(link_mode=LinkMode.COPY))

    assert [result.status for result in results] == ["synced", "synced"]
    assert len(fake.calls) == 2
