import subprocess
from pathlib import Path

import pytest

import studiolink.lmstudio_adapter as adapter_module
from studiolink.lmstudio_adapter import LMStudioAdapter, LMStudioError
from studiolink.models import LinkMode


def test_os_error_has_actionable_cli_message(make_config, monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("missing lms binary")

    monkeypatch.setattr(subprocess, "run", missing)
    with pytest.raises(LMStudioError, match="missing lms binary"):
        LMStudioAdapter(make_config()).get_version()


def test_import_failure_includes_stdout(make_config, monkeypatch):
    def failure(*args, **kwargs):
        raise subprocess.CalledProcessError(
            1, args[0], output="Target file already exists", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", failure)
    config = make_config()
    monkeypatch.setattr(
        adapter_module,
        "lmstudio_models_directory",
        lambda home: config.lmstudio_models_dir,
    )
    with pytest.raises(LMStudioError, match="Target file already exists"):
        LMStudioAdapter(config).import_model(
            "model.gguf", user_repo="ollama/test", link_mode=LinkMode.HARD_LINK
        )


@pytest.mark.parametrize("mode", list(LinkMode))
def test_import_command_preserves_requested_mode(make_config, monkeypatch, mode):
    config = make_config()
    monkeypatch.setattr(
        adapter_module,
        "lmstudio_models_directory",
        lambda home: config.lmstudio_models_dir,
    )
    commands = []

    def run(command, **kwargs):
        commands.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    LMStudioAdapter(config).import_model(
        "a model.gguf", user_repo="ollama/test", link_mode=mode, dry_run=True
    )
    command, options = commands[0]
    assert command[:2] == [str(config.lms_exe), "import"]
    assert Path(command[2]).name == "a model.gguf"
    assert command[3:5] == ["--yes", "--user-repo"]
    assert command[5].startswith(".studiolink-transactions/import-")
    assert command[6:] == [mode.lms_flag, "--dry-run"]
    assert options["timeout"] == adapter_module.IMPORT_TIMEOUT


def test_mismatched_directory_refused_before_import(make_config, monkeypatch, tmp_path):
    config = make_config()
    monkeypatch.setattr(
        adapter_module, "lmstudio_models_directory", lambda home: tmp_path / "elsewhere"
    )

    def must_not_run(*args, **kwargs):
        pytest.fail("must not import into the wrong directory")

    monkeypatch.setattr(subprocess, "run", must_not_run)
    with pytest.raises(LMStudioError, match="models directory differs"):
        LMStudioAdapter(config).import_model(
            "model.gguf", user_repo="ollama/test", link_mode=LinkMode.COPY
        )


def test_import_wraps_os_execution_error(make_config, monkeypatch):
    config = make_config()
    adapter = LMStudioAdapter(config)
    monkeypatch.setattr(
        adapter_module,
        "lmstudio_models_directory",
        lambda home: config.lmstudio_models_dir,
    )

    def missing_executable(*args, **kwargs):
        raise FileNotFoundError("lms is missing")

    monkeypatch.setattr(
        "studiolink.lmstudio_adapter.subprocess.run", missing_executable
    )

    with pytest.raises(LMStudioError, match="Could not run LM Studio CLI"):
        adapter.import_model(
            "model.gguf",
            user_repo="ollama/model",
            link_mode=LinkMode.COPY,
        )
