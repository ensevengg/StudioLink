"""Exercise imports with real files, including the LM Studio destination."""

import os
from pathlib import Path

import pytest
from conftest import DIGEST_A, write_blob, write_manifest

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
