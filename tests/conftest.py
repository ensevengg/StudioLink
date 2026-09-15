from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Make the src tree importable when running from a checkout without
# installing (CI installs the package, so this is a no-op there).
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from studiolink.config import StudioLinkConfig  # noqa: E402
from studiolink.lmstudio_adapter import LMStudioError  # noqa: E402
from studiolink.models import (  # noqa: E402
    ImportResult,
    LinkMode,
    OllamaModel,
    SyncRecord,
)

MODEL_MEDIA_TYPE = "application/vnd.ollama.image.model"
DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64

STUDIO_ENV_VARS = (
    "STUDIOLINK_OLLAMA_EXE",
    "STUDIOLINK_LMS_EXE",
    "STUDIOLINK_OLLAMA_MODELS_DIR",
    "STUDIOLINK_LMSTUDIO_MODELS_DIR",
    "STUDIOLINK_STATE_DIR",
    "OLLAMA_MODELS",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Keep tests hermetic against StudioLink/Ollama env vars on the host."""
    for name in STUDIO_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def make_config(tmp_path):
    def _make(**overrides):
        models_dir = tmp_path / "ollama" / "models"
        state_dir = tmp_path / "studiolink"
        values = dict(
            ollama_exe=tmp_path / "bin" / "ollama.exe",
            lms_exe=tmp_path / "bin" / "lms.exe",
            ollama_models_dir=models_dir,
            ollama_manifests_dir=models_dir / "manifests",
            ollama_blobs_dir=models_dir / "blobs",
            lmstudio_models_dir=tmp_path / "lmstudio" / "models",
            state_dir=state_dir,
            state_file=state_dir / "state.json",
            import_staging_dir=state_dir / "imports",
        )
        values.update(overrides)
        return StudioLinkConfig(**values)

    return _make


def blob_filename(digest: str) -> str:
    return digest.replace(":", "-")


def write_blob(config: StudioLinkConfig, digest: str, *, gguf: bool = True) -> Path:
    config.ollama_blobs_dir.mkdir(parents=True, exist_ok=True)
    path = config.ollama_blobs_dir / blob_filename(digest)
    header = b"GGUF" if gguf else b"XXXX"
    path.write_bytes(header + b"0" * 8)
    return path


def write_manifest(
    config: StudioLinkConfig,
    *,
    registry: str = "registry.ollama.ai",
    namespace: str = "library",
    repository: str = "llama3",
    tag: str = "1b",
    digest: str = DIGEST_A,
    size: int = 1234,
    content: object | None = None,
    raw_text: str | None = None,
) -> Path:
    manifest_path = (
        config.ollama_manifests_dir / registry / namespace / repository / tag
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_text is not None:
        manifest_path.write_text(raw_text, encoding="utf-8")
        return manifest_path
    if content is None:
        content = {
            "schemaVersion": 2,
            "layers": [{"mediaType": MODEL_MEDIA_TYPE, "digest": digest, "size": size}],
        }
    manifest_path.write_text(json.dumps(content), encoding="utf-8")
    return manifest_path


def make_ollama_model(
    canonical_name: str = "llama3:1b",
    digest: str | None = DIGEST_A,
    *,
    registry: str = "registry.ollama.ai",
    namespace: str = "library",
    repository: str | None = None,
    tag: str | None = None,
    gguf_valid: bool = True,
    issues: tuple[str, ...] = (),
    blob_path: Path | None = Path("blob"),
) -> OllamaModel:
    repo, _, tg = canonical_name.rpartition(":")
    return OllamaModel(
        canonical_name=canonical_name,
        fully_qualified_name=f"{registry}/{namespace}/{repo or repository}:{tg}",
        registry=registry,
        namespace=namespace,
        repository=repository or repo,
        tag=tag or tg,
        manifest_path=Path("manifest"),
        model_digest=digest,
        blob_path=blob_path,
        declared_size=1,
        blob_size=1,
        gguf_valid=gguf_valid,
        issues=tuple(issues),
    )


def make_sync_record(
    canonical_name: str = "llama3:1b", digest: str = DIGEST_A
) -> SyncRecord:
    return SyncRecord(
        canonical_name=canonical_name,
        digest=digest,
        blob_path=Path("blob"),
        import_alias_path=Path("alias"),
        user_repo="ollama/llama3",
        link_mode=LinkMode.HARD_LINK,
        imported_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


class FakeLMStudio:
    """In-memory LMStudioPort double."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.fail_for: set[str] = set()
        self.timeout_for: set[str] = set()
        self.models_dir: Path | None = None

    def get_version(self) -> str | None:
        return "0.3.17 (fake)"

    def get_import_capabilities(self) -> set[LinkMode]:
        return {LinkMode.HARD_LINK, LinkMode.COPY, LinkMode.SYMBOLIC_LINK}

    def import_model(
        self,
        source_path: str,
        *,
        user_repo: str,
        link_mode: LinkMode,
        dry_run: bool = False,
    ) -> ImportResult:
        self.calls.append(
            {
                "source_path": source_path,
                "user_repo": user_repo,
                "link_mode": link_mode,
                "dry_run": dry_run,
            }
        )
        if source_path in self.fail_for:
            raise LMStudioError(
                f"LM Studio import failed for {source_path}: boom",
                ["lms"],
                1,
                "boom",
            )
        if source_path in self.timeout_for:
            raise subprocess.TimeoutExpired(cmd=["lms"], timeout=3600)
        if not dry_run and self.models_dir is not None:
            target = self.models_dir / user_repo / Path(source_path).name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(source_path).read_bytes())
        return ImportResult(
            command=("lms", "import", source_path),
            stdout="",
            stderr="",
            return_code=0,
            dry_run=dry_run,
        )


@pytest.fixture
def fake_lmstudio() -> FakeLMStudio:
    return FakeLMStudio()


@pytest.fixture
def make_syncer(make_config, fake_lmstudio):
    from studiolink.state import StateStore
    from studiolink.syncer import Syncer

    def _make(config: StudioLinkConfig | None = None):
        cfg = config or make_config()
        fake_lmstudio.models_dir = cfg.lmstudio_models_dir
        syncer = Syncer(cfg, fake_lmstudio, StateStore(cfg.state_file))
        return syncer, cfg, fake_lmstudio

    return _make
