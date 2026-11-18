from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from studiolink.models import LinkMode


def _expand_path(env_name: str, default: Path) -> Path:
    raw = os.environ.get(env_name)
    return Path(raw).expanduser() if raw else default.expanduser()


# Where Linux Ollama installs keep models: the official install script runs the
# service as the "ollama" user with HOME=/usr/share/ollama, distro packages
# (Arch, CachyOS) set OLLAMA_MODELS=/var/lib/ollama, and a manually started
# `ollama serve` uses the invoking user's home.
LINUX_OLLAMA_MODELS_DIRS = (
    Path("/usr/share/ollama/.ollama/models"),
    Path("/var/lib/ollama"),
)


def _systemd_ollama_models_dir() -> Path | None:
    """Read OLLAMA_MODELS from the ollama systemd unit, if one sets it.

    The service's environment is invisible to the user's shell, so a
    relocated store would otherwise go unnoticed.
    """
    try:
        result = subprocess.run(
            [
                "systemctl",
                "show",
                "ollama.service",
                "--property=Environment",
                "--value",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        entries = shlex.split(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    for entry in entries:
        name, _, value = entry.partition("=")
        if name == "OLLAMA_MODELS" and value:
            return Path(value)
    return None


def _has_ollama_store(path: Path) -> bool:
    return any(
        (path / name).is_dir() for name in ("manifests-v2", "manifests", "blobs")
    )


def _has_ollama_models(path: Path) -> bool:
    """Whether the store holds at least one named manifest.

    The service creates empty store directories the first time it starts, so
    an empty store must never win over one that really holds the user's models.
    """
    for name in ("manifests-v2", "manifests"):
        root = path / name
        if root.is_dir():
            for entry in root.rglob("*"):
                if entry.is_file() or entry.is_symlink():
                    return True
    return False


def _default_ollama_models_dir(home: Path) -> Path:
    if not sys.platform.startswith("linux"):
        return home / ".ollama" / "models"
    service_dir = _systemd_ollama_models_dir()
    candidates = [
        *([service_dir] if service_dir is not None else []),
        *LINUX_OLLAMA_MODELS_DIRS,
        home / ".ollama" / "models",
    ]
    for present in (_has_ollama_models, _has_ollama_store):
        for candidate in candidates:
            try:
                if present(candidate):
                    return candidate
            except OSError:
                continue
    return candidates[0]


def _default_ollama_exe(home: Path) -> Path:
    if sys.platform == "win32":
        return Path(
            shutil.which("ollama")
            or home / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe"
        )
    return Path(shutil.which("ollama") or "/usr/bin/ollama")


def _default_lms_exe(home: Path) -> Path:
    if sys.platform == "win32":
        return Path(shutil.which("lms") or home / ".lmstudio" / "bin" / "lms.exe")
    return Path(shutil.which("lms") or home / ".lmstudio" / "bin" / "lms")


def lmstudio_models_directory(home: Path) -> Path:
    fallback = home / ".lmstudio" / "models"
    settings_path = home / ".lmstudio" / "settings.json"
    try:
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return fallback
    if not isinstance(payload, dict):
        return fallback
    downloads_folder = payload.get("downloadsFolder")
    if not isinstance(downloads_folder, str) or not downloads_folder.strip():
        return fallback
    return Path(downloads_folder).expanduser()


@dataclass(slots=True, frozen=True)
class StudioLinkConfig:
    ollama_exe: Path
    lms_exe: Path
    ollama_models_dir: Path
    ollama_manifests_dir: Path
    ollama_blobs_dir: Path
    lmstudio_models_dir: Path
    state_dir: Path
    state_file: Path
    import_staging_dir: Path
    # None picks the cheapest mode this machine supports (files.choose_link_mode).
    default_link_mode: LinkMode | None = None
    # Ollama 0.40+ writes named manifests to manifests-v2/; None derives it
    # from ollama_models_dir.
    ollama_manifests_v2_dir: Path | None = None

    @property
    def ollama_manifests_v2_path(self) -> Path:
        if self.ollama_manifests_v2_dir is not None:
            return self.ollama_manifests_v2_dir
        return self.ollama_models_dir / "manifests-v2"

    @classmethod
    def from_env(cls) -> "StudioLinkConfig":
        home = Path.home()
        # Honor Ollama's own OLLAMA_MODELS relocation variable unless the
        # StudioLink-specific override is set.
        ollama_models_dir = _expand_path(
            "STUDIOLINK_OLLAMA_MODELS_DIR",
            _expand_path("OLLAMA_MODELS", _default_ollama_models_dir(home)),
        )
        state_dir = _expand_path("STUDIOLINK_STATE_DIR", home / ".studiolink")
        return cls(
            ollama_exe=_expand_path(
                "STUDIOLINK_OLLAMA_EXE",
                _default_ollama_exe(home),
            ),
            lms_exe=_expand_path(
                "STUDIOLINK_LMS_EXE",
                _default_lms_exe(home),
            ),
            ollama_models_dir=ollama_models_dir,
            ollama_manifests_dir=ollama_models_dir / "manifests",
            ollama_manifests_v2_dir=ollama_models_dir / "manifests-v2",
            ollama_blobs_dir=ollama_models_dir / "blobs",
            lmstudio_models_dir=_expand_path(
                "STUDIOLINK_LMSTUDIO_MODELS_DIR",
                lmstudio_models_directory(home),
            ),
            state_dir=state_dir,
            state_file=state_dir / "state.json",
            import_staging_dir=state_dir / "imports",
        )
