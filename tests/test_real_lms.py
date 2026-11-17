"""Opt-in live CLI checks, isolated from the user's LM Studio by mount namespaces.

Run with STUDIOLINK_TEST_REAL_LMS=1. Linux, bubblewrap and lms are required.
No model weights are loaded; the tiny GGUF fixtures exercise importing only.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from studiolink.models import LinkMode


@pytest.fixture
def real_cli(tmp_path):
    if os.environ.get("STUDIOLINK_TEST_REAL_LMS") != "1":
        pytest.skip("set STUDIOLINK_TEST_REAL_LMS=1 for isolated real-lms tests")
    bwrap = shutil.which("bwrap")
    lms = shutil.which("lms")
    if not sys.platform.startswith("linux") or not bwrap or not lms:
        pytest.skip("real-lms sandbox requires Linux, bubblewrap and lms")
    probe = subprocess.run(
        [bwrap, "--ro-bind", "/", "/", "--unshare-net", "--", "true"],
        capture_output=True,
        text=True,
    )
    if probe.returncode:
        pytest.skip(f"bubblewrap unavailable: {probe.stderr.strip()}")
    root = tmp_path / "library with spaces"
    root.mkdir()
    lmhome = root / "lmstudio-home"
    lmhome.mkdir()
    models = root / "lmstudio-models"
    (lmhome / "settings.json").write_text(json.dumps({"downloadsFolder": str(models)}))
    binary = root / "real-lms"
    binary.touch()
    source = root / "ollama"
    state = root / "state"
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
            "STUDIOLINK_OLLAMA_MODELS_DIR": str(source),
            "STUDIOLINK_LMSTUDIO_MODELS_DIR": str(models),
            "STUDIOLINK_STATE_DIR": str(state),
            "STUDIOLINK_LMS_EXE": str(binary),
        }
    )
    command = [
        bwrap,
        "--die-with-parent",
        "--unshare-net",
        "--ro-bind",
        "/",
        "/",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--bind",
        str(root),
        str(root),
        "--ro-bind",
        lms,
        str(binary),
        "--bind",
        str(lmhome),
        str(Path.home() / ".lmstudio"),
        "--",
        sys.executable,
        "-m",
        "studiolink",
    ]

    class CLI:
        def run(self, *args, expected=0):
            completed = subprocess.run(
                [*command, *args, "--json"],
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert completed.returncode == expected, (
                completed.stdout[:2200],
                completed.stderr[:2200],
            )
            return json.loads(completed.stdout)

        def model(self, *, tag="small", payload=b""):
            content = struct.pack("<4sIQQ", b"GGUF", 3, 0, 0) + payload
            digest = "sha256:" + hashlib.sha256(content).hexdigest()
            blob = source / "blobs" / digest.replace(":", "-")
            blob.parent.mkdir(parents=True, exist_ok=True)
            blob.write_bytes(content)
            manifest = (
                source
                / "manifests"
                / "registry.ollama.ai"
                / "library"
                / "fixture"
                / tag
            )
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(
                json.dumps(
                    {
                        "layers": [
                            {
                                "mediaType": "application/vnd.ollama.image.model",
                                "digest": digest,
                                "size": len(content),
                            }
                        ]
                    }
                )
            )
            return blob, manifest

        def target(self):
            paths = list(models.rglob("*.gguf"))
            assert len(paths) == 1, paths
            return paths[0]

    cli = CLI()
    cli.root, cli.models, cli.source, cli.state = root, models, source, state
    return cli


@pytest.mark.parametrize("mode", list(LinkMode))
def test_real_lms_full_lifecycle(real_cli, mode):
    blob, manifest = real_cli.model()
    assert (
        real_cli.run("sync", "--all", "--dry-run", mode.lms_flag)[0]["status"]
        == "dry-run"
    )
    assert not list(real_cli.models.rglob("*.gguf"))
    assert real_cli.run("sync", "--all", mode.lms_flag)[0]["status"] == "synced"
    target = real_cli.target()
    assert target.read_bytes() == blob.read_bytes()
    assert target.is_symlink() == (mode is LinkMode.SYMBOLIC_LINK)
    assert target.samefile(blob) == (mode is not LinkMode.COPY)
    assert real_cli.run("sync", "--all", mode.lms_flag)[0]["status"] == "skipped"
    target.unlink()
    assert not real_cli.run("status")[0]["synced"]
    assert real_cli.run("sync", "--all", mode.lms_flag)[0]["status"] == "synced"
    (real_cli.state / "state.json").unlink()
    assert real_cli.run("sync", "--all", mode.lms_flag)[0]["status"] == "synced"
    new_blob, manifest = real_cli.model(payload=b"updated weights")
    assert real_cli.run("sync", "--all", mode.lms_flag)[0]["status"] == "synced"
    assert not target.exists() and not target.is_symlink()
    target = real_cli.target()
    assert target.read_bytes() == new_blob.read_bytes()
    manifest.unlink()
    new_blob.unlink()
    real_cli.run("prune", "--dry-run")
    assert target.is_file()
    real_cli.run("prune")
    assert not list(real_cli.models.rglob("*.gguf"))
    assert not list((real_cli.state / "imports").glob("*.gguf"))
    assert json.loads((real_cli.state / "state.json").read_text())["sync_records"] == {}


@pytest.mark.parametrize("first", list(LinkMode))
@pytest.mark.parametrize("second", list(LinkMode))
@pytest.mark.parametrize("lose_state", [False, True])
def test_real_lms_mode_conversion(real_cli, first, second, lose_state):
    blob, _ = real_cli.model()
    real_cli.run("sync", "--all", first.lms_flag)
    if lose_state:
        (real_cli.state / "state.json").unlink()
    real_cli.run("sync", "--all", second.lms_flag)
    target = real_cli.target()
    assert target.read_bytes() == blob.read_bytes()
    assert target.is_symlink() == (second is LinkMode.SYMBOLIC_LINK)
    assert target.samefile(blob) == (second is not LinkMode.COPY)


@pytest.mark.parametrize("initial", [None, LinkMode.HARD_LINK, LinkMode.SYMBOLIC_LINK])
def test_real_lms_identical_tags_use_one_copy(real_cli, initial):
    real_cli.model(tag="one")
    real_cli.model(tag="two")
    if initial is not None:
        real_cli.run("sync", "--all", initial.lms_flag)
    assert all(
        row["status"] == "synced" for row in real_cli.run("sync", "--all", "--copy")
    )
    targets = list(real_cli.models.rglob("*.gguf"))
    assert len(targets) == 2
    assert targets[0].samefile(targets[1])


@pytest.mark.parametrize("mode", list(LinkMode))
def test_real_lms_direct_import(real_cli, mode):
    blob, manifest = real_cli.model()
    results = real_cli.run("sync", "--all", "--direct", mode.lms_flag)
    assert results[0]["status"] == "synced"
    target = real_cli.models / "ollama/fixture" / blob.name
    assert target.read_bytes() == blob.read_bytes()
    assert target.is_symlink() == (mode is LinkMode.SYMBOLIC_LINK)
    assert target.samefile(blob) == (mode is not LinkMode.COPY)
    assert not (real_cli.state / "imports").exists()
    manifest.unlink()
    real_cli.run("prune")
    assert not target.exists() and not target.is_symlink()
    assert blob.is_file()
