import errno
import os

from conftest import (
    DIGEST_A,
    DIGEST_B,
    make_ollama_model,
    write_blob,
    write_manifest,
)

import studiolink.syncer as syncer_module
from studiolink.lmstudio_adapter import LMStudioError
from studiolink.models import (
    ArtifactRole,
    ImportMode,
    ImportResult,
    LinkMode,
    ModelReadiness,
)
from studiolink.ollama_adapter import OllamaAdapter
from studiolink.state import StateStore
from studiolink.syncer import (
    _ensure_import_alias,
    _format_import_message,
    same_file,
)


def scan_models(config):
    return OllamaAdapter(config).scan_models()


def sync_kwargs(**overrides):
    defaults = dict(
        link_mode=LinkMode.HARD_LINK,
        import_mode=ImportMode.ALIAS,
        dry_run=False,
    )
    defaults.update(overrides)
    return defaults


class TestHappyPath:
    def test_sync_creates_alias_and_record(self, make_syncer):
        syncer, config, fake = make_syncer()
        write_manifest(config, repository="llama3", tag="1b", digest=DIGEST_A)
        write_blob(config, DIGEST_A)
        model = scan_models(config)[0]

        results = syncer.sync([model], **sync_kwargs())

        assert results[0].status == "synced"
        alias = config.import_staging_dir / model.import_filename
        assert alias.exists()
        assert same_file(alias, model.blob_path)
        assert fake.calls[0]["source_path"] == str(alias)
        assert fake.calls[0]["user_repo"] == model.user_repo
        record = StateStore(config.state_file).get_record("llama3:1b")
        assert record is not None and record.digest == DIGEST_A
        assert record.import_alias_path == alias
        assert record.imported_model_path == (
            config.lmstudio_models_dir / model.user_repo / model.import_filename
        )

    def test_second_sync_skips(self, make_syncer):
        syncer, config, fake = make_syncer()
        write_manifest(config, digest=DIGEST_A)
        write_blob(config, DIGEST_A)

        first = syncer.sync(scan_models(config), **sync_kwargs())
        second = syncer.sync(scan_models(config), **sync_kwargs())

        assert first[0].status == "synced"
        assert second[0].status == "skipped"
        assert len(fake.calls) == 1

    def test_direct_mode_imports_blob_path_without_alias(self, make_syncer):
        syncer, config, fake = make_syncer()
        write_manifest(config, digest=DIGEST_A)
        write_blob(config, DIGEST_A)
        model = scan_models(config)[0]

        results = syncer.sync([model], **sync_kwargs(import_mode=ImportMode.DIRECT))

        assert results[0].status == "synced"
        assert fake.calls[0]["source_path"] == str(model.blob_path)
        assert not config.import_staging_dir.exists()


class TestProjectorBundle:
    def test_sync_imports_model_and_projector_into_one_repository(self, make_syncer):
        syncer, config, fake = make_syncer()
        projector_digest = "sha256:" + "c" * 64
        write_manifest(
            config,
            repository="llava",
            content={
                "layers": [
                    {
                        "mediaType": "application/vnd.ollama.image.model",
                        "digest": DIGEST_A,
                    },
                    {
                        "mediaType": "application/vnd.ollama.image.projector",
                        "digest": projector_digest,
                    },
                ]
            },
        )
        write_blob(config, DIGEST_A)
        write_blob(config, projector_digest)
        model = scan_models(config)[0]

        result = syncer.sync([model], **sync_kwargs())[0]

        assert result.status == "synced"
        assert len(fake.calls) == 2
        assert {call["user_repo"] for call in fake.calls} == {model.import_user_repo}
        assert fake.calls[1]["source_path"].endswith(
            model.artifact_import_filename(model.artifacts[1])
        )
        assert "/mmproj-" in fake.calls[1]["source_path"].replace("\\", "/")
        record = StateStore(config.state_file).get_record(model.canonical_name)
        assert record is not None
        assert [artifact.role for artifact in record.artifacts] == [
            ArtifactRole.MODEL,
            ArtifactRole.PROJECTOR,
        ]
        assert all(
            artifact.imported_model_path is not None
            and artifact.imported_model_path.is_file()
            for artifact in record.artifacts
        )

    def test_waits_for_lm_studio_to_index_projector(self, make_syncer, monkeypatch):
        syncer, config, fake = make_syncer()
        projector_digest = "sha256:" + "c" * 64
        write_manifest(
            config,
            repository="llava",
            content={
                "layers": [
                    {
                        "mediaType": "application/vnd.ollama.image.model",
                        "digest": DIGEST_A,
                    },
                    {
                        "mediaType": "application/vnd.ollama.image.projector",
                        "digest": projector_digest,
                    },
                ]
            },
        )
        write_blob(config, DIGEST_A)
        write_blob(config, projector_digest)
        model = scan_models(config)[0]
        real_list = fake.list_models
        calls = 0

        def delayed_inventory():
            nonlocal calls
            calls += 1
            return () if calls == 1 else real_list()

        monkeypatch.setattr(fake, "list_models", delayed_inventory)
        monkeypatch.setattr(syncer_module.time, "sleep", lambda _: None)

        result = syncer.sync([model], **sync_kwargs())[0]

        assert result.status == "synced"
        assert calls == 2

    def test_resync_rechecks_projector_inventory(self, make_syncer, monkeypatch):
        syncer, config, fake = make_syncer()
        projector_digest = "sha256:" + "c" * 64
        write_manifest(
            config,
            repository="llava",
            content={
                "layers": [
                    {
                        "mediaType": "application/vnd.ollama.image.model",
                        "digest": DIGEST_A,
                    },
                    {
                        "mediaType": "application/vnd.ollama.image.projector",
                        "digest": projector_digest,
                    },
                ]
            },
        )
        write_blob(config, DIGEST_A)
        write_blob(config, projector_digest)
        model = scan_models(config)[0]
        assert syncer.sync([model], **sync_kwargs())[0].status == "synced"
        fake.inventory_override = ()
        monkeypatch.setattr(syncer_module.time, "sleep", lambda _: None)

        result = syncer.sync([model], **sync_kwargs())[0]

        assert result.status == "error"
        assert "vision-capable" in result.message

    def test_inventory_failure_is_reported_for_model(self, make_syncer, monkeypatch):
        syncer, config, fake = make_syncer()
        projector_digest = "sha256:" + "c" * 64
        write_manifest(
            config,
            repository="llava",
            content={
                "layers": [
                    {
                        "mediaType": "application/vnd.ollama.image.model",
                        "digest": DIGEST_A,
                    },
                    {
                        "mediaType": "application/vnd.ollama.image.projector",
                        "digest": projector_digest,
                    },
                ]
            },
        )
        write_blob(config, DIGEST_A)
        write_blob(config, projector_digest)
        model = scan_models(config)[0]

        def fail_inventory():
            raise LMStudioError("inventory unavailable", ["lms"], 1, "boom")

        monkeypatch.setattr(fake, "list_models", fail_inventory)

        result = syncer.sync([model], **sync_kwargs())[0]

        assert result.status == "error"
        assert "inventory unavailable" in result.message
        record = StateStore(config.state_file).get_record(model.canonical_name)
        assert record is not None
        assert record.vision_confirmed is False

    def test_projector_failure_saves_progress_and_retry_resumes(self, make_syncer):
        syncer, config, fake = make_syncer()
        projector_digest = "sha256:" + "c" * 64
        write_manifest(
            config,
            repository="llava",
            content={
                "layers": [
                    {
                        "mediaType": "application/vnd.ollama.image.model",
                        "digest": DIGEST_A,
                    },
                    {
                        "mediaType": "application/vnd.ollama.image.projector",
                        "digest": projector_digest,
                    },
                ]
            },
        )
        write_blob(config, DIGEST_A)
        write_blob(config, projector_digest)
        model = scan_models(config)[0]
        projector_alias = config.import_staging_dir / model.artifact_import_filename(
            model.artifacts[1]
        )
        fake.fail_for = {str(projector_alias)}

        first = syncer.sync([model], **sync_kwargs())[0]

        assert first.status == "error"
        partial = StateStore(config.state_file).get_record(model.canonical_name)
        assert partial is not None
        assert [artifact.role for artifact in partial.artifacts] == [ArtifactRole.MODEL]

        fake.fail_for.clear()
        second = syncer.sync([model], **sync_kwargs())[0]

        assert second.status == "synced"
        assert len(fake.calls) == 3
        assert fake.calls[-1]["source_path"] == str(projector_alias)
        complete = StateStore(config.state_file).get_record(model.canonical_name)
        assert complete is not None and len(complete.artifacts) == 2
        assert complete.vision_confirmed is True

    def test_missing_projector_prevents_every_import(self, make_syncer):
        syncer, config, fake = make_syncer()
        projector_digest = "sha256:" + "c" * 64
        write_manifest(
            config,
            repository="llava",
            content={
                "layers": [
                    {
                        "mediaType": "application/vnd.ollama.image.model",
                        "digest": DIGEST_A,
                    },
                    {
                        "mediaType": "application/vnd.ollama.image.projector",
                        "digest": projector_digest,
                    },
                ]
            },
        )
        write_blob(config, DIGEST_A)
        model = scan_models(config)[0]

        result = syncer.sync([model], **sync_kwargs())[0]

        assert result.status == "error"
        assert "ollama pull llava:1b" in result.message
        assert fake.calls == []

    def test_direct_mode_rejects_projector_bundle(self, make_syncer):
        syncer, config, fake = make_syncer()
        projector_digest = "sha256:" + "c" * 64
        write_manifest(
            config,
            repository="llava",
            content={
                "layers": [
                    {
                        "mediaType": "application/vnd.ollama.image.model",
                        "digest": DIGEST_A,
                    },
                    {
                        "mediaType": "application/vnd.ollama.image.projector",
                        "digest": projector_digest,
                    },
                ]
            },
        )
        write_blob(config, DIGEST_A)
        write_blob(config, projector_digest)

        result = syncer.sync(
            scan_models(config),
            **sync_kwargs(import_mode=ImportMode.DIRECT),
        )[0]

        assert result.status == "error"
        assert "--direct is not supported" in result.message
        assert fake.calls == []


class TestCrossVolumeCopy:
    def test_copy_stages_on_ollama_volume_not_state_volume(
        self, make_syncer, monkeypatch
    ):
        syncer, config, _ = make_syncer()
        write_manifest(config, digest=DIGEST_A)
        write_blob(config, DIGEST_A)
        model = scan_models(config)[0]
        real_link = os.link

        def cross_volume_link(source, target, *args, **kwargs):
            if config.import_staging_dir in target.parents:
                raise OSError(errno.EXDEV, "Invalid cross-device link")
            return real_link(source, target, *args, **kwargs)

        monkeypatch.setattr("studiolink.syncer.os.link", cross_volume_link)

        results = syncer.sync([model], **sync_kwargs(link_mode=LinkMode.COPY))

        assert results[0].status == "synced"
        assert results[0].record is not None
        assert results[0].record.imported_model_path.is_file()
        assert not (config.import_staging_dir / model.import_filename).exists()

    def test_copy_removes_reused_temporary_alias(self, make_syncer):
        syncer, config, _ = make_syncer()
        write_manifest(config, digest=DIGEST_A)
        blob = write_blob(config, DIGEST_A)
        model = scan_models(config)[0]
        staging = config.ollama_models_dir / ".studiolink-imports"
        staging.mkdir(parents=True)
        alias = staging / model.import_filename
        os.link(blob, alias)

        results = syncer.sync([model], **sync_kwargs(link_mode=LinkMode.COPY))

        assert results[0].status == "synced"
        assert not alias.exists()


class TestUnreadyModels:
    def test_stale_blob_errors_with_pull_hint(self, make_syncer):
        syncer, config, fake = make_syncer()
        write_manifest(config, digest=DIGEST_A)  # no blob written

        results = syncer.sync(scan_models(config), **sync_kwargs())

        assert results[0].status == "error"
        assert "ollama pull llama3:1b" in results[0].message
        assert fake.calls == []
        assert not results[0].record

    def test_invalid_model_errors(self, make_syncer):
        syncer, config, _ = make_syncer()
        model = make_ollama_model(
            gguf_valid=False,
            issues=("blob does not start with GGUF magic bytes",),
            blob_path=None,
        )
        results = syncer.sync([model], **sync_kwargs())
        assert results[0].status == "error"
        assert "not ready for import" in results[0].message


class TestDryRun:
    def test_dry_run_cleans_alias_and_skips_state(self, make_syncer):
        syncer, config, fake = make_syncer()
        write_manifest(config, digest=DIGEST_A)
        write_blob(config, DIGEST_A)
        model = scan_models(config)[0]

        results = syncer.sync([model], **sync_kwargs(dry_run=True))

        assert results[0].status == "dry-run"
        assert fake.calls[0]["dry_run"] is True
        assert not (config.import_staging_dir / model.import_filename).exists()
        assert StateStore(config.state_file).get_all_records() == {}

    def test_dry_run_failure_still_cleans_alias(self, make_syncer):
        syncer, config, fake = make_syncer()
        write_manifest(config, digest=DIGEST_A)
        write_blob(config, DIGEST_A)
        model = scan_models(config)[0]
        fake.fail_for = {str(config.import_staging_dir / model.import_filename)}

        results = syncer.sync([model], **sync_kwargs(dry_run=True))

        assert results[0].status == "error"
        assert not (config.import_staging_dir / model.import_filename).exists()


class TestFailureIsolation:
    def test_one_failure_does_not_abort_batch(self, make_syncer):
        # Regression: an LMStudioError used to propagate and abort the
        # remaining models in a `sync --all` batch.
        syncer, config, fake = make_syncer()
        write_manifest(config, repository="llama3", tag="1b", digest=DIGEST_A)
        write_blob(config, DIGEST_A)
        write_manifest(config, repository="mistral", tag="7b", digest=DIGEST_B)
        write_blob(config, DIGEST_B)
        models = {m.canonical_name: m for m in scan_models(config)}
        fake.fail_for = {
            str(config.import_staging_dir / models["llama3:1b"].import_filename)
        }

        results = syncer.sync(
            [models["llama3:1b"], models["mistral:7b"]], **sync_kwargs()
        )

        assert [r.status for r in results] == ["error", "synced"]
        assert "boom" in results[0].message
        records = StateStore(config.state_file).get_all_records()
        assert set(records) == {"mistral:7b"}

    def test_timeout_is_isolated_per_model(self, make_syncer):
        syncer, config, fake = make_syncer()
        write_manifest(config, digest=DIGEST_A)
        write_blob(config, DIGEST_A)
        model = scan_models(config)[0]
        fake.timeout_for = {str(config.import_staging_dir / model.import_filename)}

        results = syncer.sync([model], **sync_kwargs())

        assert results[0].status == "error"
        assert "timed out" in results[0].message


class TestAliasReuse:
    def test_repull_relinks_alias_to_new_blob(self, make_syncer):
        # Regression for the silent-stale-import bug: after re-pulling a
        # model under the same name, the import must use the NEW blob.
        syncer, config, fake = make_syncer()
        write_manifest(config, repository="llama3", tag="1b", digest=DIGEST_A)
        blob_a = write_blob(config, DIGEST_A)

        first = syncer.sync(scan_models(config), **sync_kwargs())
        assert first[0].status == "synced"
        alias_a = config.import_staging_dir / first[0].record.import_alias_path.name
        assert same_file(alias_a, blob_a)

        # Ollama re-pulls the same tag with new content.
        write_manifest(config, repository="llama3", tag="1b", digest=DIGEST_B)
        blob_b = write_blob(config, DIGEST_B)
        model = scan_models(config)[0]
        assert model.import_filename != first[0].record.import_alias_path.name

        second = syncer.sync([model], **sync_kwargs())

        assert second[0].status == "synced"
        alias_b = config.import_staging_dir / model.import_filename
        assert alias_b.exists()
        assert same_file(alias_b, blob_b)
        assert fake.calls[-1]["source_path"] == str(alias_b)

    def test_stale_alias_file_is_replaced(self, make_config):
        config = make_config()
        write_manifest(config, digest=DIGEST_A)
        write_blob(config, DIGEST_A)
        model = scan_models(config)[0]

        config.import_staging_dir.mkdir(parents=True)
        alias = config.import_staging_dir / model.import_filename
        alias.write_bytes(b"not a hard link to the blob")

        alias_path, created = _ensure_import_alias(model, config.import_staging_dir)

        assert created is True
        assert alias_path == alias
        assert same_file(alias, model.blob_path)

    def test_matching_alias_is_reused(self, make_config):
        config = make_config()
        write_manifest(config, digest=DIGEST_A)
        write_blob(config, DIGEST_A)
        model = scan_models(config)[0]

        config.import_staging_dir.mkdir(parents=True)
        alias = config.import_staging_dir / model.import_filename
        os.link(model.blob_path, alias)

        alias_path, created = _ensure_import_alias(model, config.import_staging_dir)

        assert created is False
        assert alias_path == alias

    def test_missing_blob_raises(self, make_config):
        config = make_config()
        model = make_ollama_model(blob_path=None)
        try:
            _ensure_import_alias(model, config.import_staging_dir)
        except RuntimeError as exc:
            assert "No blob path" in str(exc)
        else:
            raise AssertionError("expected RuntimeError")


class TestImportMessage:
    def test_uses_last_output_line(self):
        result = ImportResult(
            command=("lms",),
            stdout="Importing...\nCopied 3 files",
            stderr="",
            return_code=0,
        )
        assert _format_import_message(result) == "Copied 3 files"

    def test_falls_back_to_stderr(self):
        result = ImportResult(
            command=("lms",), stdout="", stderr="warned", return_code=0
        )
        assert _format_import_message(result) == "warned"

    def test_default_message(self):
        result = ImportResult(command=("lms",), stdout="", stderr="", return_code=0)
        assert _format_import_message(result) == "imported successfully"


class TestStateDurability:
    def test_record_saved_after_each_model(self, make_syncer):
        # A crash mid-batch must not lose already-imported models.
        syncer, config, fake = make_syncer()
        write_manifest(config, repository="llama3", tag="1b", digest=DIGEST_A)
        write_blob(config, DIGEST_A)
        write_manifest(config, repository="mistral", tag="7b", digest=DIGEST_B)
        write_blob(config, DIGEST_B)
        models = {m.canonical_name: m for m in scan_models(config)}
        fake.fail_for = {
            str(config.import_staging_dir / models["llama3:1b"].import_filename)
        }

        syncer.sync([models["llama3:1b"], models["mistral:7b"]], **sync_kwargs())

        # Even though the batch "failed" on model 1, model 2's record persisted
        # after its own import.
        assert set(StateStore(config.state_file).get_all_records()) == {"mistral:7b"}


class TestReadinessOfSyncedModels:
    def test_sync_result_carries_model(self, make_syncer):
        syncer, config, _ = make_syncer()
        write_manifest(config, digest=DIGEST_A)
        write_blob(config, DIGEST_A)
        model = scan_models(config)[0]
        results = syncer.sync([model], **sync_kwargs())
        assert results[0].model is model
        assert results[0].model.readiness is ModelReadiness.READY
