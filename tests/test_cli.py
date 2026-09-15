import json
import logging
import subprocess
import sys

import pytest
from conftest import DIGEST_A, make_ollama_model

from studiolink import __version__, cli
from studiolink.models import (
    DoctorCheck,
    ImportMode,
    LinkMode,
    PruneReport,
    SyncResult,
)
from studiolink.service import StatusEntry


class FakeService:
    def __init__(self, config):
        self.config = config
        self.calls = []
        self._model = make_ollama_model("llama3:1b", DIGEST_A)

    def scan(self):
        self.calls.append(("scan",))
        return [self._model]

    def status(self):
        self.calls.append(("status",))
        return [StatusEntry(model=self._model, synced=False, sync_record=None)]

    def sync(
        self,
        *,
        model_names=None,
        sync_all=False,
        link_mode=None,
        import_mode=ImportMode.ALIAS,
        dry_run=False,
    ):
        self.calls.append(
            ("sync", model_names, sync_all, link_mode, import_mode, dry_run)
        )
        if not model_names and not sync_all:
            raise ValueError("provide at least one model name or use --all")
        return [SyncResult(model=self._model, status="synced", message="ok")]

    def doctor(self):
        self.calls.append(("doctor",))
        return [DoctorCheck("demo check", True, "all good")]

    def prune(self, *, dry_run=False):
        self.calls.append(("prune", dry_run))
        return PruneReport(dry_run=dry_run)


@pytest.fixture
def fake_service(monkeypatch, make_config):
    service = FakeService(make_config())
    monkeypatch.setattr(cli, "StudioLinkService", lambda *a, **k: service)
    return service


def out(capsys):
    return capsys.readouterr().out


def err(capsys):
    captured = capsys.readouterr()
    return captured.out + captured.err


class TestParserMatrix:
    def test_no_args_prints_help(self, fake_service, capsys):
        assert cli.main([]) == 0
        assert "usage" in out(capsys)

    def test_help_command(self, fake_service, capsys):
        # Regression: `sdl help` used to exit 1 with "Invalid command".
        assert cli.main(["help"]) == 0
        text = out(capsys)
        assert "Available Commands" in text
        assert "prune" in text

    def test_version_flag(self, fake_service, capsys):
        with pytest.raises(SystemExit) as exit_info:
            cli.main(["--version"])
        assert exit_info.value.code == 0
        assert __version__ in out(capsys)

    def test_command_help_flag(self, fake_service, capsys):
        with pytest.raises(SystemExit) as exit_info:
            cli.main(["sync", "--help"])
        assert exit_info.value.code == 0
        assert "--dry-run" in out(capsys)

    def test_invalid_command_exits_nonzero(self, fake_service, capsys):
        with pytest.raises(SystemExit) as exit_info:
            cli.main(["badcmd"])
        assert exit_info.value.code == 2
        assert "invalid choice" in err(capsys)

    def test_verbose_flag_after_subcommand(self, fake_service, caplog):
        # Regression: `sdl scan -v` used to fail with "Invalid command".
        caplog.set_level(logging.DEBUG, logger="studiolink")
        assert cli.main(["scan", "-v"]) == 0
        debug_records = [
            r
            for r in caplog.records
            if r.name == "studiolink" and r.levelno == logging.DEBUG
        ]
        assert debug_records, "expected debug logging to be enabled"

    def test_verbose_flag_before_subcommand(self, fake_service, caplog):
        caplog.set_level(logging.DEBUG, logger="studiolink")
        assert cli.main(["-v", "scan"]) == 0
        debug_records = [
            r
            for r in caplog.records
            if r.name == "studiolink" and r.levelno == logging.DEBUG
        ]
        assert debug_records, "expected debug logging to be enabled"

    def test_no_verbose_means_no_debug(self, fake_service, caplog):
        caplog.set_level(logging.DEBUG, logger="studiolink")
        assert cli.main(["scan"]) == 0
        debug_records = [
            r
            for r in caplog.records
            if r.name == "studiolink" and r.levelno == logging.DEBUG
        ]
        assert not debug_records

    def test_verbose_long_form(self, fake_service, caplog):
        caplog.set_level(logging.DEBUG, logger="studiolink")
        assert cli.main(["scan", "--verbose"]) == 0
        assert any(
            r.name == "studiolink" and r.levelno == logging.DEBUG
            for r in caplog.records
        )


class TestSyncCommand:
    def test_sync_requires_models(self, fake_service, capsys):
        assert cli.main(["sync"]) == 2
        assert "provide at least one model name" in err(capsys)

    def test_link_mode_conflict_rejected(self, fake_service, capsys):
        assert cli.main(["sync", "llama3", "--copy", "--hard-link"]) == 2
        assert "choose only one" in err(capsys)

    def test_sync_passes_arguments_through(self, fake_service):
        assert cli.main(["sync", "llama3:1b", "--dry-run", "--copy"]) == 0
        call = fake_service.calls[-1]
        assert call[0] == "sync"
        assert call[1] == ["llama3:1b"]
        assert call[3] is LinkMode.COPY
        assert call[5] is True

    def test_sync_direct_flag(self, fake_service):
        assert cli.main(["sync", "llama3:1b", "--direct"]) == 0
        call = fake_service.calls[-1]
        assert call[4] is ImportMode.DIRECT

    def test_sync_json_output(self, fake_service, capsys):
        assert cli.main(["sync", "llama3:1b", "--json"]) == 0
        payload = json.loads(out(capsys))
        assert payload[0]["status"] == "synced"
        assert payload[0]["model"] == "llama3:1b"

    def test_sync_error_result_exits_nonzero(self, fake_service, monkeypatch, capsys):
        def failing_sync(**kwargs):
            return [
                SyncResult(model=fake_service._model, status="error", message="bad")
            ]

        monkeypatch.setattr(fake_service, "sync", failing_sync)
        assert cli.main(["sync", "llama3:1b"]) == 1


class TestJsonOutput:
    def test_scan_json(self, fake_service, capsys):
        assert cli.main(["scan", "--json"]) == 0
        payload = json.loads(out(capsys))
        assert payload[0]["canonical_name"] == "llama3:1b"
        assert payload[0]["readiness"] == "ready"
        assert payload[0]["artifacts"] == []

    def test_status_json(self, fake_service, capsys):
        assert cli.main(["status", "--json"]) == 0
        payload = json.loads(out(capsys))
        assert payload[0]["status"] == "pending"
        assert payload[0]["synced"] is False

    def test_doctor_json(self, fake_service, capsys):
        assert cli.main(["doctor", "--json"]) == 0
        payload = json.loads(out(capsys))
        assert payload[0]["ok"] is True

    def test_doctor_failure_exits_nonzero(self, fake_service, monkeypatch):
        monkeypatch.setattr(
            fake_service,
            "doctor",
            lambda: [DoctorCheck("broken", False, "nope")],
        )
        assert cli.main(["doctor"]) == 1


class TestPruneCommand:
    def test_prune_registered(self, fake_service, capsys):
        assert cli.main(["prune"]) == 0
        assert fake_service.calls[-1] == ("prune", False)
        assert "Nothing to prune" in out(capsys)

    def test_prune_dry_run(self, fake_service):
        assert cli.main(["prune", "--dry-run"]) == 0
        assert fake_service.calls[-1] == ("prune", True)

    def test_prune_json(self, fake_service, capsys):
        assert cli.main(["prune", "--json"]) == 0
        payload = json.loads(out(capsys))
        assert payload == {"dry_run": False, "aliases": [], "records_removed": []}

    def test_prune_reports_aliases(self, fake_service, monkeypatch, capsys):
        from pathlib import Path

        from studiolink.models import PruneResult

        report = PruneReport(
            dry_run=False,
            aliases=(PruneResult(Path("old.gguf"), 1024, True, "model gone", True),),
            records_removed=("old:model",),
        )
        monkeypatch.setattr(fake_service, "prune", lambda *, dry_run=False: report)
        assert cli.main(["prune"]) == 0
        text = out(capsys)
        assert "old.gguf" in text
        assert "1024 bytes" in text
        assert "old:model" in text


class TestVersionCompare:
    def test_multi_digit_patch_compares_numerically(self):
        # Regression: string comparison said 0.1.10 <= 0.1.9.
        assert cli._version_tuple("0.1.10") > cli._version_tuple("0.1.9")
        assert cli._version_tuple("0.2.0") > cli._version_tuple("0.1.9")
        assert cli._version_tuple("1.0.0") > cli._version_tuple("0.99.99")

    def test_versions_with_suffixes_do_not_crash(self):
        assert cli._version_tuple("0.1.0rc1") == (0, 1, 1)


class TestUpgradeCommand:
    def test_network_failure_exits_nonzero(self, fake_service, monkeypatch, capsys):
        def broken_urlopen(url, timeout=10):
            raise OSError("no network")

        monkeypatch.setattr(cli.urllib.request, "urlopen", broken_urlopen)
        assert cli.main(["upgrade"]) == 1
        assert "Could not check for updates" in err(capsys)

    def test_already_on_latest(self, fake_service, monkeypatch, capsys):
        class FakeResponse:
            def read(self):
                return json.dumps({"info": {"version": "0.0.1"}}).encode()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        monkeypatch.setattr(
            cli.urllib.request, "urlopen", lambda url, timeout=10: FakeResponse()
        )
        assert cli.main(["upgrade"]) == 0
        assert "Already on latest" in out(capsys)

    def test_upgrade_invokes_module_pip(self, fake_service, monkeypatch, capsys):
        class FakeResponse:
            def read(self):
                return json.dumps({"info": {"version": "99.0.0"}}).encode()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        commands = []

        def fake_run(command, **kwargs):
            commands.append(command)
            return subprocess.CompletedProcess(command, 0, "", "")

        monkeypatch.setattr(
            cli.urllib.request, "urlopen", lambda url, timeout=10: FakeResponse()
        )
        monkeypatch.setattr(cli.subprocess, "run", fake_run)

        assert cli.main(["upgrade"]) == 0

        assert commands, "expected pip to be invoked"
        command = commands[0]
        assert command[0] == sys.executable
        assert command[1:3] == ["-m", "pip"]
        assert "--upgrade" in command
        assert "Upgraded" in out(capsys)

    def test_upgrade_failure_shows_pip_output(self, fake_service, monkeypatch, capsys):
        class FakeResponse:
            def read(self):
                return json.dumps({"info": {"version": "99.0.0"}}).encode()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def failing_run(command, **kwargs):
            return subprocess.CompletedProcess(
                command, 1, "", "externally-managed-environment"
            )

        monkeypatch.setattr(
            cli.urllib.request, "urlopen", lambda url, timeout=10: FakeResponse()
        )
        monkeypatch.setattr(cli.subprocess, "run", failing_run)

        assert cli.main(["upgrade"]) == 1
        text = err(capsys)
        assert "externally-managed-environment" in text
        assert "virtualenv" in text
