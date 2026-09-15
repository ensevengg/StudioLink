from conftest import DIGEST_A, DIGEST_B, make_ollama_model, make_sync_record

from studiolink.models import (
    LinkMode,
    ModelReadiness,
    PruneReport,
    PruneResult,
    is_synced,
)


class TestImportFilename:
    def test_contains_hash_portion_of_digest(self):
        digest = "sha256:" + "abc123def456" + "0" * 52
        model = make_ollama_model(canonical_name="llama3.2:1b", digest=digest)
        assert model.import_filename == "llama3.2-1b-abc123def456.gguf"

    def test_different_digests_produce_different_filenames(self):
        # Regression: the old [:5] truncation yielded the constant "sha25"
        # for every model, so re-pulled models collided on one alias.
        model_a = make_ollama_model(canonical_name="llama3.2:1b", digest=DIGEST_A)
        model_b = make_ollama_model(canonical_name="llama3.2:1b", digest=DIGEST_B)
        assert model_a.import_filename != model_b.import_filename

    def test_filename_uses_hash_not_algorithm_prefix(self):
        model = make_ollama_model(canonical_name="m:1", digest="sha256:" + "f" * 64)
        assert model.import_filename.endswith("f" * 12 + ".gguf")
        assert "sha25" not in model.import_filename

    def test_unknown_digest_marker(self):
        model = make_ollama_model(canonical_name="mymodel:latest", digest=None)
        assert model.import_filename == "mymodel-latest-unknown.gguf"

    def test_sanitizes_unsafe_characters(self):
        model = make_ollama_model(
            canonical_name="we ird$model:x", digest="sha256:" + "c" * 64
        )
        name = model.import_filename
        assert name == "we-ird-model-x-" + "c" * 12 + ".gguf"
        for ch in name:
            assert ch.isalnum() or ch in ("-", "_", ".")

    def test_collapses_repeated_dashes(self):
        model = make_ollama_model(
            canonical_name="a$$b:latest", digest="sha256:" + "d" * 64
        )
        assert "--" not in model.import_filename


class TestUserRepo:
    def test_library_repository(self):
        assert make_ollama_model("llama3:1b").user_repo == "ollama/llama3"

    def test_namespaced_repository(self):
        model = make_ollama_model("gemma:2b", namespace="marella", repository="gemma")
        assert model.user_repo == "ollama/marella-gemma"

    def test_custom_registry_dots_replaced(self):
        model = make_ollama_model(
            "gemma:2b",
            registry="ghcr.io",
            namespace="team",
            repository="gemma",
        )
        assert model.user_repo == "ollama/ghcr-io-team-gemma"


class TestReadiness:
    def test_ready(self):
        assert make_ollama_model("m:1").readiness is ModelReadiness.READY

    def test_stale_when_blob_missing(self):
        model = make_ollama_model(
            "m:1",
            gguf_valid=False,
            issues=("model blob is missing from the Ollama blob store",),
        )
        assert model.readiness is ModelReadiness.STALE

    def test_invalid_when_bad_magic(self):
        model = make_ollama_model(
            "m:1",
            gguf_valid=False,
            issues=("blob does not start with GGUF magic bytes",),
        )
        assert model.readiness is ModelReadiness.INVALID


class TestIsSynced:
    def test_no_record(self):
        assert is_synced(make_ollama_model(), None) is False

    def test_matching_digest(self):
        model = make_ollama_model("llama3:1b", DIGEST_A)
        assert is_synced(model, make_sync_record("llama3:1b", DIGEST_A)) is True

    def test_digest_mismatch_triggers_resync(self):
        model = make_ollama_model("llama3:1b", DIGEST_B)
        assert is_synced(model, make_sync_record("llama3:1b", DIGEST_A)) is False


class TestSyncRecordRoundtrip:
    def test_to_from_json(self):
        record = make_sync_record("llama3:1b", DIGEST_A)
        restored = type(record).from_json(record.to_json())
        assert restored == record

    def test_import_command_defaults_empty(self):
        restored = make_sync_record().to_json()
        assert restored["import_command"] == []


class TestPruneTypes:
    def test_prune_result_json(self):
        from pathlib import Path

        result = PruneResult(
            path=Path("alias.gguf"),
            size=10,
            would_free=True,
            reason="gone",
            removed=True,
        )
        assert result.to_json() == {
            "path": str(Path("alias.gguf")),
            "size": 10,
            "would_free": True,
            "reason": "gone",
            "removed": True,
        }

    def test_prune_report_json(self):
        from pathlib import Path

        report = PruneReport(
            dry_run=True,
            aliases=(PruneResult(Path("a"), 1, True, "r", False),),
            records_removed=("old:model",),
        )
        payload = report.to_json()
        assert payload["dry_run"] is True
        assert payload["records_removed"] == ["old:model"]
        assert payload["aliases"][0]["removed"] is False


class TestLinkModeFlag:
    def test_lms_flags(self):
        assert LinkMode.HARD_LINK.lms_flag == "--hard-link"
        assert LinkMode.COPY.lms_flag == "--copy"
        assert LinkMode.SYMBOLIC_LINK.lms_flag == "--symbolic-link"
