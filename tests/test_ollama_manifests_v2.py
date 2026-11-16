"""Ollama 0.40+ ``manifests-v2`` layout and manifest-list selection.

The fixtures mirror what Ollama itself writes: manifest JSON is stored as a
content-addressed blob and ``manifests-v2/<host>/<ns>/<model>/<tag>`` is a
relative symlink to it (a regular-file copy on Windows). Manifest lists hold
per-runner children that reference further manifest blobs.
"""

from __future__ import annotations

import hashlib
import json
import os

import pytest
from conftest import (
    DIGEST_A,
    DIGEST_B,
    MODEL_MEDIA_TYPE,
    make_sync_record,
    write_blob,
    write_manifest,
)

from studiolink.config import StudioLinkConfig
from studiolink.models import ModelReadiness
from studiolink.ollama_adapter import (
    MANIFEST_LIST_MEDIA_TYPE,
    MANIFEST_MEDIA_TYPE,
    MLX_ONLY_ISSUE,
    OllamaAdapter,
)
from studiolink.service import StudioLinkService
from studiolink.state import StateStore

DIGEST_C = "sha256:" + "c" * 64
TENSOR_MEDIA_TYPE = "application/vnd.ollama.image.tensor"


def gguf_manifest(digest: str, **extra: object) -> dict[str, object]:
    return {
        "schemaVersion": 2,
        "mediaType": MANIFEST_MEDIA_TYPE,
        "layers": [{"mediaType": MODEL_MEDIA_TYPE, "digest": digest, "size": 12}],
        **extra,
    }


def store_manifest_blob(config: StudioLinkConfig, document: object) -> str:
    data = json.dumps(document).encode("utf-8")
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    config.ollama_blobs_dir.mkdir(parents=True, exist_ok=True)
    (config.ollama_blobs_dir / digest.replace(":", "-")).write_bytes(data)
    return digest


def name_v2(
    config: StudioLinkConfig,
    digest: str,
    *,
    host: str = "ollama.com",
    namespace: str = "library",
    repository: str = "llama3",
    tag: str = "1b",
    symlink: bool = True,
):
    path = config.ollama_manifests_v2_path / host / namespace / repository / tag
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = config.ollama_blobs_dir / digest.replace(":", "-")
    if not symlink:
        path.write_bytes(blob.read_bytes())
        return path
    try:
        os.symlink(os.path.relpath(blob, path.parent), path)
    except (OSError, NotImplementedError) as exc:  # pragma: no cover - Windows
        pytest.skip(f"symlinks unavailable: {exc}")
    return path


def write_v2(config: StudioLinkConfig, document: object, **name: object):
    return name_v2(config, store_manifest_blob(config, document), **name)


def scan(config: StudioLinkConfig):
    return OllamaAdapter(config).scan_report()


def only_model(config: StudioLinkConfig):
    report = scan(config)
    assert report.complete, report.errors
    assert len(report.models) == 1, report.models
    return report.models[0]


class TestV2Names:
    def test_v2_only_model_is_discovered_with_legacy_identity(self, make_config):
        config = make_config()
        write_blob(config, DIGEST_A)
        path = write_v2(config, gguf_manifest(DIGEST_A))

        model = only_model(config)

        assert model.readiness is ModelReadiness.READY
        assert model.canonical_name == "llama3:1b"
        assert model.registry == "registry.ollama.ai"
        assert model.fully_qualified_name == "registry.ollama.ai/library/llama3:1b"
        assert model.user_repo == "ollama/llama3"
        assert model.manifest_path == path
        assert model.model_digest == DIGEST_A

    def test_regular_file_copy_is_supported(self, make_config):
        # Windows (0.40.1+) copies the manifest blob instead of linking it.
        config = make_config()
        write_blob(config, DIGEST_A)
        write_v2(config, gguf_manifest(DIGEST_A), symlink=False)

        assert only_model(config).readiness is ModelReadiness.READY

    def test_v2_wins_over_legacy_for_the_same_name(self, make_config):
        config = make_config()
        write_blob(config, DIGEST_A)
        write_blob(config, DIGEST_B)
        write_manifest(config, digest=DIGEST_A)
        write_v2(config, gguf_manifest(DIGEST_B))

        model = only_model(config)

        assert model.model_digest == DIGEST_B

    def test_dangling_v2_link_falls_back_to_legacy(self, make_config):
        config = make_config()
        write_blob(config, DIGEST_A)
        write_manifest(config, digest=DIGEST_A)
        path = write_v2(config, gguf_manifest(DIGEST_B))
        os.remove(os.path.realpath(path))

        model = only_model(config)

        assert model.model_digest == DIGEST_A
        assert model.readiness is ModelReadiness.READY

    def test_dangling_v2_link_alone_is_stale_not_hidden(self, make_config):
        config = make_config()
        path = write_v2(config, gguf_manifest(DIGEST_A))
        os.remove(os.path.realpath(path))

        model = only_model(config)

        assert model.canonical_name == "llama3:1b"
        assert model.readiness is ModelReadiness.STALE

    def test_legacy_and_v2_models_are_merged_and_sorted(self, make_config):
        config = make_config()
        write_blob(config, DIGEST_A)
        write_manifest(config, repository="zeta", digest=DIGEST_A)
        write_v2(config, gguf_manifest(DIGEST_A), repository="alpha")

        names = [m.canonical_name for m in scan(config).models]

        assert names == ["alpha:1b", "zeta:1b"]

    def test_v2_only_library_is_available(self, make_config):
        config = make_config()
        write_blob(config, DIGEST_A)
        write_v2(config, gguf_manifest(DIGEST_A))

        report = scan(config)

        assert not config.ollama_manifests_dir.exists()
        assert report.source_available is True

    def test_encoded_port_in_host_is_decoded(self, make_config):
        config = make_config()
        write_blob(config, DIGEST_A)
        write_v2(
            config,
            gguf_manifest(DIGEST_A),
            host="localhost%3A5000",
            namespace="team",
            repository="coder",
        )

        model = only_model(config)

        assert model.registry == "localhost:5000"
        assert model.canonical_name == "localhost:5000/team/coder:1b"

    def test_converted_legacy_shadow_tags_are_ignored(self, make_config):
        config = make_config()
        write_blob(config, DIGEST_A)
        write_manifest(config, digest=DIGEST_A)
        write_manifest(config, repository="llamacpp", tag="d" * 64, digest=DIGEST_A)

        assert only_model(config).canonical_name == "llama3:1b"

    def test_legacy_downgrade_anchor_still_parses(self, make_config):
        # Ollama appends manifest/list references to the legacy entry it
        # keeps for downgrades; they are not model artifacts.
        config = make_config()
        write_blob(config, DIGEST_A)
        anchor = gguf_manifest(DIGEST_A)
        anchor["layers"] += [
            {"mediaType": MANIFEST_LIST_MEDIA_TYPE, "digest": DIGEST_B, "size": 1},
            {"mediaType": MANIFEST_MEDIA_TYPE, "digest": DIGEST_C, "size": 1},
        ]
        write_manifest(config, content=anchor)

        assert only_model(config).readiness is ModelReadiness.READY

    def test_link_whose_content_does_not_match_its_digest_is_invalid(self, make_config):
        config = make_config()
        write_blob(config, DIGEST_A)
        path = write_v2(config, gguf_manifest(DIGEST_A))
        with open(os.path.realpath(path), "a", encoding="utf-8") as handle:
            handle.write(" ")

        model = only_model(config)

        assert model.readiness is ModelReadiness.INVALID
        assert "manifest link does not match its content digest" in model.issues


def child_ref(config, document, runner, fmt="gguf"):
    return {
        "mediaType": MANIFEST_MEDIA_TYPE,
        "digest": store_manifest_blob(config, document),
        "runner": runner,
        "format": fmt,
    }


def manifest_list(*children):
    return {
        "schemaVersion": 2,
        "mediaType": MANIFEST_LIST_MEDIA_TYPE,
        "manifests": list(children),
    }


class TestManifestLists:
    def test_prefers_llamacpp_child_over_ggml(self, make_config):
        config = make_config()
        write_blob(config, DIGEST_A)
        write_blob(config, DIGEST_B)
        write_v2(
            config,
            manifest_list(
                child_ref(config, gguf_manifest(DIGEST_A), "ggml"),
                child_ref(config, gguf_manifest(DIGEST_B), "llamacpp"),
            ),
        )

        model = only_model(config)

        assert model.readiness is ModelReadiness.READY
        assert model.model_digest == DIGEST_B

    def test_uses_ggml_child_when_it_is_the_only_gguf(self, make_config):
        config = make_config()
        write_blob(config, DIGEST_A)
        write_v2(
            config,
            manifest_list(child_ref(config, gguf_manifest(DIGEST_A), "ggml")),
        )

        assert only_model(config).model_digest == DIGEST_A

    def test_falls_back_when_preferred_child_blob_is_missing(self, make_config):
        config = make_config()
        write_blob(config, DIGEST_A)
        missing = child_ref(config, gguf_manifest(DIGEST_B), "llamacpp")
        (config.ollama_blobs_dir / missing["digest"].replace(":", "-")).unlink()
        write_v2(
            config,
            manifest_list(missing, child_ref(config, gguf_manifest(DIGEST_A), "ggml")),
        )

        assert only_model(config).model_digest == DIGEST_A

    def test_inline_children_are_supported(self, make_config):
        config = make_config()
        write_blob(config, DIGEST_A)
        child = gguf_manifest(DIGEST_A, runner="llamacpp", format="gguf")
        write_v2(config, manifest_list(child))

        assert only_model(config).model_digest == DIGEST_A

    def test_missing_gguf_child_blob_is_stale(self, make_config):
        config = make_config()
        child = child_ref(config, gguf_manifest(DIGEST_A), "llamacpp")
        (config.ollama_blobs_dir / child["digest"].replace(":", "-")).unlink()
        write_v2(config, manifest_list(child))

        model = only_model(config)

        assert model.readiness is ModelReadiness.STALE

    def test_mlx_only_list_is_reported_clearly(self, make_config):
        config = make_config()
        mlx = {
            "schemaVersion": 2,
            "mediaType": MANIFEST_MEDIA_TYPE,
            "layers": [{"mediaType": TENSOR_MEDIA_TYPE, "digest": DIGEST_A}],
        }
        write_v2(config, manifest_list(child_ref(config, mlx, "mlx", "safetensors")))

        model = only_model(config)

        assert model.readiness is ModelReadiness.INVALID
        assert model.issues == (MLX_ONLY_ISSUE,)

    def test_plain_mlx_manifest_is_reported_clearly(self, make_config):
        config = make_config()
        write_v2(
            config,
            {
                "schemaVersion": 2,
                "mediaType": MANIFEST_MEDIA_TYPE,
                "runner": "mlx",
                "format": "safetensors",
                "layers": [{"mediaType": TENSOR_MEDIA_TYPE, "digest": DIGEST_A}],
            },
        )

        model = only_model(config)

        assert model.readiness is ModelReadiness.INVALID
        assert model.issues == (MLX_ONLY_ISSUE,)

    def test_child_blob_with_wrong_content_is_invalid(self, make_config):
        config = make_config()
        write_blob(config, DIGEST_A)
        child = child_ref(config, gguf_manifest(DIGEST_A), "llamacpp")
        blob = config.ollama_blobs_dir / child["digest"].replace(":", "-")
        blob.write_bytes(blob.read_bytes() + b" ")
        write_v2(config, manifest_list(child))

        model = only_model(config)

        assert model.readiness is ModelReadiness.INVALID
        assert any("does not match its digest" in issue for issue in model.issues)


class TestV2Lifecycle:
    def test_prune_keeps_import_for_model_only_in_v2(self, make_config, fake_lmstudio):
        # Regression: models that exist only in manifests-v2 (e.g. after
        # `ollama cp`) looked deleted, so prune removed their LM Studio import.
        config = make_config()
        config.lmstudio_models_dir.mkdir(parents=True)
        write_blob(config, DIGEST_A)
        write_v2(config, gguf_manifest(DIGEST_A))
        store = StateStore(config.state_file)
        store.upsert(make_sync_record("llama3:1b", DIGEST_A))
        service = StudioLinkService(
            config=config, ollama=OllamaAdapter(config), lmstudio=fake_lmstudio
        )

        report = service.prune()

        assert report.records_removed == ()
        assert set(store.get_all_records()) == {"llama3:1b"}

    def test_doctor_accepts_a_v2_only_library(self, make_config, fake_lmstudio):
        config = make_config()
        config.ollama_manifests_v2_path.mkdir(parents=True)
        service = StudioLinkService(
            config=config, ollama=OllamaAdapter(config), lmstudio=fake_lmstudio
        )

        check = next(c for c in service.doctor() if c.name == "ollama manifests dir")

        assert check.ok is True
        assert "manifests-v2" in check.details
