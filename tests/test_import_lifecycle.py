"""Exercise imports with real files, including the LM Studio destination."""

import os
import shutil
from pathlib import Path

import pytest
from conftest import DIGEST_A, DIGEST_B, write_blob, write_manifest

from studiolink.models import ImportMode, LinkMode
from studiolink.service import StudioLinkService


def destination(config, model):
    return config.lmstudio_models_dir / model.user_repo / model.import_filename


def test_deleted_destination_is_detected_and_repaired(library):
    config, service, adapter, _, _ = library
    service.sync(sync_all=True)
    target = destination(config, service.scan()[0])
    target.unlink()
    assert not service.status()[0].synced
    assert service.sync(sync_all=True)[0].status == "synced"
    assert target.is_file()


def test_changing_link_mode_is_not_silently_skipped(library):
    config, service, _, _, blob = library
    service.sync(sync_all=True, link_mode=LinkMode.COPY)
    result = service.sync(sync_all=True, link_mode=LinkMode.HARD_LINK)
    assert result[0].status == "synced"
    assert destination(config, service.scan()[0]).samefile(blob)


def test_failed_import_does_not_pin_blob(library):
    config, service, adapter, _, blob = library
    model = service.scan()[0]
    alias = config.import_staging_dir / model.import_filename
    adapter.fail_for.add(str(alias))
    assert service.sync(sync_all=True)[0].status == "error"
    assert not alias.exists()
    assert blob.stat().st_nlink == 1


def test_repull_does_not_leave_old_imports(library):
    config, service, _, _, old_blob = library
    service.sync(sync_all=True)
    old_target = destination(config, service.scan()[0])
    old_alias = config.import_staging_dir / service.scan()[0].import_filename
    write_manifest(config, digest=DIGEST_B)
    write_blob(config, DIGEST_B)
    assert service.sync(sync_all=True)[0].status == "synced"
    assert not old_target.exists()
    assert not old_alias.exists()
    assert old_blob.stat().st_nlink == 1


@pytest.mark.parametrize("mode", list(LinkMode))
def test_prune_removes_managed_destination_and_state(library, mode):
    config, service, _, manifest, blob = library
    service.sync(sync_all=True, link_mode=mode)
    target = destination(config, service.scan()[0])
    manifest.unlink()
    blob.unlink()
    service.prune()
    assert not target.exists() and not target.is_symlink()
    assert service.state.get_all_records() == {}
    assert list(config.import_staging_dir.glob("*.gguf")) == []


def test_unavailable_library_does_not_destroy_aliases(library):
    config, service, _, _, _ = library
    service.sync(sync_all=True, link_mode=LinkMode.SYMBOLIC_LINK)
    model = service.scan()[0]
    target = destination(config, model)
    shutil.rmtree(config.ollama_models_dir)
    try:
        service.prune()
    except (ValueError, OSError, RuntimeError):
        pass
    assert target.is_file()


def test_sync_recovers_after_state_loss_without_reimport(library):
    config, service, adapter, _, _ = library
    service.sync(sync_all=True)
    config.state_file.unlink()
    assert service.sync(sync_all=True)[0].status != "error"
    assert len(adapter.calls) == 1
    assert service.state.get_all_records()


@pytest.mark.parametrize("first", list(LinkMode))
@pytest.mark.parametrize("second", list(LinkMode))
def test_state_loss_and_mode_change_recovers_verified_import(library, first, second):
    config, service, _, _, blob = library
    assert service.sync(sync_all=True, link_mode=first)[0].status == "synced"
    config.state_file.unlink()
    result = service.sync(sync_all=True, link_mode=second)[0]
    assert result.status == "synced", result.message
    target = destination(config, service.scan()[0])
    assert target.read_bytes() == blob.read_bytes()
    assert target.is_symlink() == (second is LinkMode.SYMBOLIC_LINK)
    assert target.samefile(blob) == (second is not LinkMode.COPY)


def test_state_loss_does_not_adopt_unrelated_file(library):
    config, service, _, _, _ = library
    service.sync(sync_all=True, link_mode=LinkMode.COPY)
    target = destination(config, service.scan()[0])
    target.write_bytes(b"other content")
    config.state_file.unlink()
    assert service.sync(sync_all=True)[0].status == "error"
    assert target.read_bytes() == b"other content"


def test_converting_identical_tags_to_copy_reuses_one_copy(library):
    config, service, _, _, blob = library
    write_manifest(config, tag="another")
    service.sync(sync_all=True)
    results = service.sync(sync_all=True, link_mode=LinkMode.COPY)
    assert all(result.status == "synced" for result in results)
    targets = [destination(config, model) for model in service.scan()]
    assert len(targets) == 2
    assert targets[0].samefile(targets[1])
    assert not targets[0].samefile(blob)


def test_direct_tags_sharing_a_target_can_change_to_copy(library):
    config, service, _, _, blob = library
    write_manifest(config, tag="another")
    service.sync(sync_all=True, import_mode=ImportMode.DIRECT)
    results = service.sync(
        sync_all=True, import_mode=ImportMode.DIRECT, link_mode=LinkMode.COPY
    )
    assert all(result.status == "synced" for result in results), [
        result.message for result in results
    ]
    records = list(service.state.get_all_records().values())
    assert records[0].imported_model_path == records[1].imported_model_path
    assert all(entry.synced for entry in service.status())
    target = records[0].imported_model_path
    assert target.read_bytes() == blob.read_bytes()
    assert not target.samefile(blob)


def test_failed_mode_conversion_preserves_working_import(library, monkeypatch):
    config, service, _, _, blob = library
    service.sync(sync_all=True, link_mode=LinkMode.COPY)
    target = destination(config, service.scan()[0])
    before = target.read_bytes()

    def fail_link(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(os, "link", fail_link)
    result = service.sync(sync_all=True, link_mode=LinkMode.HARD_LINK)
    assert result[0].status == "error"
    assert target.read_bytes() == before
    assert not target.samefile(blob)
    assert list(target.parent.glob(".studiolink-*")) == []


def test_prune_does_not_delete_replaced_destination(library):
    config, service, _, manifest, blob = library
    service.sync(sync_all=True)
    target = destination(config, service.scan()[0])
    target.unlink()
    target.write_bytes(b"An unrelated replacement")
    manifest.unlink()
    blob.unlink()
    service.prune()
    assert target.read_bytes() == b"An unrelated replacement"
    assert service.state.get_all_records()


def test_prune_dry_run_counts_shared_inode_once(library):
    config, service, _, manifest, blob = library
    service.sync(sync_all=True)
    target = destination(config, service.scan()[0])
    expected_size = blob.stat().st_size
    manifest.unlink()
    blob.unlink()
    report = service.prune(dry_run=True)
    assert target.is_file()
    assert (
        sum(p.size for p in (*report.aliases, *report.imports) if p.would_free)
        == expected_size
    )
    report = service.prune()
    assert (
        sum(
            p.size
            for p in (*report.aliases, *report.imports)
            if p.would_free and p.removed
        )
        == expected_size
    )


def test_copy_can_stage_across_filesystems_without_extra_copy(library, monkeypatch):
    import errno

    config, service, _, _, blob = library
    original = os.link

    def cross_volume(source, target, *args, **kwargs):
        if Path(target).parent == config.import_staging_dir:
            raise OSError(errno.EXDEV, "cross-device link")
        return original(source, target, *args, **kwargs)

    monkeypatch.setattr(os, "link", cross_volume)
    assert service.sync(sync_all=True, link_mode=LinkMode.COPY)[0].status == "synced"
    model = service.scan()[0]
    assert not (config.import_staging_dir / model.import_filename).exists()
    assert not (config.ollama_models_dir / ".studiolink-imports").exists()
    assert destination(config, model).read_bytes() == blob.read_bytes()


def test_deleted_blob_is_not_reported_synced(library):
    _, service, _, _, blob = library
    service.sync(sync_all=True)
    blob.unlink()
    assert service.status()[0].display_status == "stale"
    assert service.sync(sync_all=True)[0].status == "error"


def test_replaced_blob_same_digest_is_relinked(library):
    config, service, _, _, blob = library
    service.sync(sync_all=True)
    blob.unlink()
    write_blob(config, DIGEST_A)
    assert not service.status()[0].synced
    assert service.sync(sync_all=True)[0].status == "synced"
    assert destination(config, service.scan()[0]).samefile(blob)


def test_copy_tags_share_one_destination_allocation(library):
    config, service, _, _, blob = library
    write_manifest(config, tag="latest")
    assert all(
        r.status == "synced"
        for r in service.sync(sync_all=True, link_mode=LinkMode.COPY)
    )
    targets = [destination(config, m) for m in service.scan()]
    assert targets[0].samefile(targets[1])
    assert not targets[0].samefile(blob)


def test_dry_run_does_not_replace_existing_alias(library):
    config, service, _, _, blob = library
    service.sync(sync_all=True)
    model = service.scan()[0]
    alias = config.import_staging_dir / model.import_filename
    original_inode = alias.stat().st_ino
    blob.unlink()
    write_blob(config, DIGEST_A)
    service.sync(sync_all=True, dry_run=True)
    assert alias.stat().st_ino == original_inode


def test_sync_and_prune_do_not_overlap(library):
    _, service, adapter, _, _ = library
    with service.state.lock():
        with pytest.raises(RuntimeError, match="Another StudioLink"):
            service.sync(sync_all=True)
        with pytest.raises(RuntimeError, match="Another StudioLink"):
            service.prune()
    assert adapter.calls == []
    assert service.sync(sync_all=True)[0].status == "synced"


def test_repull_keeps_alias_shared_by_other_model(library):
    config, service, _, original_manifest, _ = library
    original_manifest.unlink()
    write_manifest(config, repository="a-b", tag="c")
    write_manifest(config, repository="a", tag="b-c")
    service.sync(sync_all=True, link_mode=LinkMode.SYMBOLIC_LINK)
    other = next(m for m in service.scan() if m.canonical_name == "a:b-c")
    target = destination(config, other)
    write_manifest(config, repository="a-b", tag="c", digest=DIGEST_B)
    write_blob(config, DIGEST_B)
    assert (
        service.sync(model_names=["a-b:c"], link_mode=LinkMode.SYMBOLIC_LINK)[0].status
        == "synced"
    )
    assert target.is_file()


def test_failed_old_cleanup_can_be_retried_without_new_copy(library, monkeypatch):
    config, service, adapter, _, _ = library
    service.sync(sync_all=True, link_mode=LinkMode.COPY)
    old_target = destination(config, service.scan()[0])
    write_manifest(config, digest=DIGEST_B)
    write_blob(config, DIGEST_B)
    real_unlink = Path.unlink

    def deny_old(path, *args, **kwargs):
        if path == old_target:
            raise PermissionError("busy")
        return real_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", deny_old)
        assert service.sync(sync_all=True, link_mode=LinkMode.COPY)[0].status == "error"
    assert service.sync(sync_all=True, link_mode=LinkMode.COPY)[0].status == "synced"
    assert len(adapter.calls) == 2


@pytest.mark.parametrize("mode", [LinkMode.HARD_LINK, LinkMode.COPY])
def test_sync_recovers_after_models_directory_is_moved(library, mode):
    from dataclasses import replace

    config, service, adapter, _, _ = library
    service.sync(sync_all=True, link_mode=mode)
    relocated = config.lmstudio_models_dir.parent / "relocated"
    config.lmstudio_models_dir.rename(relocated)
    new_config = replace(config, lmstudio_models_dir=relocated)
    adapter.config = new_config
    new_service = StudioLinkService(config=new_config, lmstudio=adapter)
    assert new_service.status()[0].synced
    assert new_service.sync(sync_all=True, link_mode=mode)[0].status == "skipped"
    assert len(adapter.calls) == 1
