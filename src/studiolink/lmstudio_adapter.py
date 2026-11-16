from __future__ import annotations

import json
import subprocess
from pathlib import Path

from studiolink.config import StudioLinkConfig, lmstudio_models_directory
from studiolink.files import destination_path, healthy_destination
from studiolink.models import ImportResult, LinkMode, LMStudioModel
from studiolink.transactions import TRANSACTIONS, import_transaction, publish_import

DEFAULT_TIMEOUT = 60
# Importing with --copy moves multi-GB GGUF files; give it an hour instead of
# the short timeout used for version/capability probes.
IMPORT_TIMEOUT = 3600


class LMStudioError(Exception):
    """Raised when LM Studio CLI command fails."""

    def __init__(
        self, message: str, command: list[str], return_code: int, stderr: str
    ) -> None:
        super().__init__(message)
        self.command = command
        self.return_code = return_code
        self.stderr = stderr


class LMStudioAdapter:
    def __init__(self, config: StudioLinkConfig) -> None:
        self.config = config

    def get_version(self) -> str | None:
        try:
            result = self._run([str(self.config.lms_exe), "--version"], check=True)
        except subprocess.CalledProcessError as exc:
            raise LMStudioError(
                f"Failed to get LM Studio version: {exc.stderr}",
                exc.cmd,
                exc.returncode,
                exc.stderr,
            ) from exc
        output = "\n".join(part for part in (result.stdout, result.stderr) if part)
        return output.strip() or None

    def get_import_capabilities(self) -> set[LinkMode]:
        try:
            result = self._run(
                [str(self.config.lms_exe), "import", "--help"], check=True
            )
        except subprocess.CalledProcessError as exc:
            raise LMStudioError(
                f"Failed to get LM Studio import capabilities: {exc.stderr}",
                exc.cmd,
                exc.returncode,
                exc.stderr,
            ) from exc
        help_text = "\n".join(part for part in (result.stdout, result.stderr) if part)
        capabilities: set[LinkMode] = set()
        if "--hard-link" in help_text:
            capabilities.add(LinkMode.HARD_LINK)
        if "--copy" in help_text:
            capabilities.add(LinkMode.COPY)
        if "--symbolic-link" in help_text:
            capabilities.add(LinkMode.SYMBOLIC_LINK)
        return capabilities

    def list_models(self) -> tuple[LMStudioModel, ...]:
        command = [str(self.config.lms_exe), "ls", "--json"]
        try:
            result = self._run(command, check=True)
        except subprocess.CalledProcessError as exc:
            raise LMStudioError(
                f"Failed to list LM Studio models: {exc.stderr}",
                exc.cmd,
                exc.returncode,
                exc.stderr,
            ) from exc
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise LMStudioError(
                f"LM Studio returned invalid model inventory JSON: {exc}",
                command,
                result.returncode,
                result.stderr,
            ) from exc
        if not isinstance(payload, list):
            raise LMStudioError(
                "LM Studio model inventory was not a JSON list",
                command,
                result.returncode,
                result.stderr,
            )
        inventory: list[LMStudioModel] = []
        for item in payload:
            if not isinstance(item, dict):
                raise LMStudioError(
                    "LM Studio model inventory contained an invalid entry",
                    command,
                    result.returncode,
                    result.stderr,
                )
            inventory.append(
                LMStudioModel(
                    model_key=str(item.get("modelKey", "")),
                    path=str(item.get("path", "")),
                    vision=item.get("vision") is True,
                    model_type=str(item.get("type", "")),
                )
            )
        return tuple(inventory)

    def import_model(
        self,
        source_path: str,
        *,
        user_repo: str,
        link_mode: LinkMode,
        dry_run: bool = False,
    ) -> ImportResult:
        actual_directory = lmstudio_models_directory(Path.home())
        if actual_directory.resolve() != self.config.lmstudio_models_dir.resolve():
            raise LMStudioError(
                "StudioLink's models directory differs from LM Studio's models directory "
                f"({actual_directory}); update LM Studio's models setting or "
                "STUDIOLINK_LMSTUDIO_MODELS_DIR before importing",
                [],
                -1,
                "models directory mismatch",
            )
        source = Path(source_path).absolute()
        target = destination_path(
            self.config.lmstudio_models_dir, user_repo, source.name
        )
        if target.exists() or target.is_symlink():
            raise LMStudioError(
                f"Import destination already exists: {target}",
                [],
                -1,
                "destination exists",
            )
        with import_transaction(self.config.lmstudio_models_dir, source.name) as staged:
            command = [
                str(self.config.lms_exe),
                "import",
                str(source),
                "--yes",
                "--user-repo",
                f"{TRANSACTIONS}/{staged.parent.name}",
                link_mode.lms_flag,
            ]
            if dry_run:
                command.append("--dry-run")
            try:
                result = self._run(command, check=True, timeout=IMPORT_TIMEOUT)
            except subprocess.CalledProcessError as exc:
                raise LMStudioError(
                    f"LM Studio import failed for {source_path}: {exc.stderr or exc.stdout}",
                    exc.cmd,
                    exc.returncode,
                    exc.stderr,
                ) from exc
            if not dry_run:
                if not healthy_destination(staged, source, link_mode):
                    raise LMStudioError(
                        f"LM Studio produced an invalid {link_mode} import",
                        command,
                        -1,
                        "invalid imported file",
                    )
                publish_import(staged, target)
            return ImportResult(
                command=tuple(command),
                stdout=f"Imported model to {target}" if not dry_run else result.stdout,
                stderr=result.stderr,
                return_code=result.returncode,
                dry_run=dry_run,
            )

    @staticmethod
    def _run(
        command: list[str], *, check: bool, timeout: int = DEFAULT_TIMEOUT
    ) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=check,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise LMStudioError(
                f"LM Studio command timed out after {timeout}s: {' '.join(command)}",
                command,
                -1,
                str(exc),
            ) from exc
        except OSError as exc:
            raise LMStudioError(
                f"Could not run LM Studio CLI {command[0]}: {exc}",
                command,
                -1,
                str(exc),
            ) from exc
