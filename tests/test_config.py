import json
import sys
from pathlib import Path

import studiolink.config as config_module
from studiolink.config import StudioLinkConfig


class TestEnvOverrides:
    def test_studiolink_models_dir_override(self, monkeypatch, tmp_path):
        custom = tmp_path / "custom-models"
        monkeypatch.setenv("STUDIOLINK_OLLAMA_MODELS_DIR", str(custom))
        config = StudioLinkConfig.from_env()
        assert config.ollama_models_dir == custom
        assert config.ollama_manifests_dir == custom / "manifests"
        assert config.ollama_blobs_dir == custom / "blobs"

    def test_state_dir_override_derives_paths(self, monkeypatch, tmp_path):
        custom = tmp_path / "state"
        monkeypatch.setenv("STUDIOLINK_STATE_DIR", str(custom))
        config = StudioLinkConfig.from_env()
        assert config.state_dir == custom
        assert config.state_file == custom / "state.json"
        assert config.import_staging_dir == custom / "imports"

    def test_exe_overrides(self, monkeypatch, tmp_path):
        monkeypatch.setenv("STUDIOLINK_OLLAMA_EXE", str(tmp_path / "ollama"))
        monkeypatch.setenv("STUDIOLINK_LMS_EXE", str(tmp_path / "lms"))
        config = StudioLinkConfig.from_env()
        assert config.ollama_exe == tmp_path / "ollama"
        assert config.lms_exe == tmp_path / "lms"


class TestOllamaModelsFallback:
    def test_ollama_models_env_used_as_default(self, monkeypatch, tmp_path):
        relocated = tmp_path / "big-disk" / "models"
        monkeypatch.setenv("OLLAMA_MODELS", str(relocated))
        config = StudioLinkConfig.from_env()
        assert config.ollama_models_dir == relocated

    def test_studiolink_var_wins_over_ollama_var(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path / "ollama-default"))
        monkeypatch.setenv("STUDIOLINK_OLLAMA_MODELS_DIR", str(tmp_path / "studio"))
        config = StudioLinkConfig.from_env()
        assert config.ollama_models_dir == tmp_path / "studio"


class TestLMStudioModelsDirectory:
    def test_uses_downloads_folder_from_lmstudio_settings(self, monkeypatch, tmp_path):
        custom = tmp_path / "large-drive" / "lm-models"
        settings = tmp_path / ".lmstudio" / "settings.json"
        settings.parent.mkdir(parents=True)
        settings.write_text(
            json.dumps({"downloadsFolder": str(custom)}), encoding="utf-8"
        )
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

        config = StudioLinkConfig.from_env()

        assert config.lmstudio_models_dir == custom

    def test_explicit_override_wins_over_lmstudio_settings(self, monkeypatch, tmp_path):
        configured = tmp_path / "settings-models"
        override = tmp_path / "override-models"
        settings = tmp_path / ".lmstudio" / "settings.json"
        settings.parent.mkdir(parents=True)
        settings.write_text(
            json.dumps({"downloadsFolder": str(configured)}), encoding="utf-8"
        )
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        monkeypatch.setenv("STUDIOLINK_LMSTUDIO_MODELS_DIR", str(override))

        config = StudioLinkConfig.from_env()

        assert config.lmstudio_models_dir == override


class TestPlatformDefaults:
    def test_windows_defaults(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(config_module.shutil, "which", lambda name: None)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        config = StudioLinkConfig.from_env()
        assert config.ollama_exe == (
            tmp_path / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe"
        )
        assert config.lms_exe == tmp_path / ".lmstudio" / "bin" / "lms.exe"

    def test_windows_prefers_ollama_on_path(self, monkeypatch, tmp_path):
        found = tmp_path / "tools" / "ollama.exe"
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(
            config_module.shutil,
            "which",
            lambda name: str(found) if name == "ollama" else None,
        )
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

        config = StudioLinkConfig.from_env()

        assert config.ollama_exe == found

    def test_windows_prefers_lms_on_path(self, monkeypatch, tmp_path):
        found = tmp_path / "tools" / "lms.exe"
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(
            config_module.shutil,
            "which",
            lambda name: str(found) if name == "lms" else None,
        )
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

        config = StudioLinkConfig.from_env()

        assert config.lms_exe == found

    def test_posix_defaults_without_path_lookup(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(config_module.shutil, "which", lambda name: None)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        config = StudioLinkConfig.from_env()
        assert config.ollama_exe == Path("/usr/bin/ollama")
        assert config.lms_exe == tmp_path / ".lmstudio" / "bin" / "lms"

    def test_posix_prefers_executables_on_path(self, monkeypatch, tmp_path):
        found = {
            "ollama": "/usr/local/bin/ollama",
            "lms": "/home/user/.local/bin/lms",
        }
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(config_module.shutil, "which", lambda name: found.get(name))
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        config = StudioLinkConfig.from_env()
        assert config.ollama_exe == Path("/usr/local/bin/ollama")
        assert config.lms_exe == Path("/home/user/.local/bin/lms")

    def test_linux_default_models_dirs(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

        config = StudioLinkConfig.from_env()

        assert config.ollama_models_dir == Path("/usr/share/ollama/.ollama/models")
        assert config.lmstudio_models_dir == tmp_path / ".lmstudio" / "models"
        assert config.state_dir == tmp_path / ".studiolink"

    def test_macos_default_models_dir(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

        config = StudioLinkConfig.from_env()

        assert config.ollama_models_dir == tmp_path / ".ollama" / "models"
