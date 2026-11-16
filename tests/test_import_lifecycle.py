"""Exercise imports with real files, including the LM Studio destination."""

import pytest


def destination(config, model):
    return config.lmstudio_models_dir / model.user_repo / model.import_filename


def test_sync_and_prune_do_not_overlap(library):
    _, service, adapter, _, _ = library
    with service.state.lock():
        with pytest.raises(RuntimeError, match="Another StudioLink"):
            service.sync(sync_all=True)
        with pytest.raises(RuntimeError, match="Another StudioLink"):
            service.prune()
    assert adapter.calls == []
    assert service.sync(sync_all=True)[0].status == "synced"
