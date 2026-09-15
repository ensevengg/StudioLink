from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import urllib.request
from datetime import timezone

from studiolink import __version__
from studiolink.lmstudio_adapter import LMStudioError
from studiolink.models import DoctorCheck, ImportMode, LinkMode, SyncResult
from studiolink.service import StatusEntry, StudioLinkService

logger = logging.getLogger("studiolink")


def _common_parser() -> argparse.ArgumentParser:
    """Options accepted both before and after the subcommand.

    The SUPPRESS default is essential: parents= shares action objects
    between the top-level parser and every subparser, and any real default
    set on the shared action would let a subcommand's parse clobber a value
    given before the subcommand (sdl -v scan).
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Enable verbose output (debug logging).",
    )
    return common


def build_parser() -> argparse.ArgumentParser:
    common = _common_parser()
    parser = argparse.ArgumentParser(
        prog="sdl",
        parents=[common],
        description="Sync Ollama-downloaded GGUF models into LM Studio.",
        epilog="Run 'sdl <command> --help' for command-specific options.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"StudioLink {__version__}",
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, metavar="<command>"
    )

    scan_parser = subparsers.add_parser(
        "scan",
        parents=[common],
        help="Discover GGUF-backed Ollama models.",
    )
    scan_parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    scan_parser.set_defaults(func=run_scan)

    sync_parser = subparsers.add_parser(
        "sync",
        parents=[common],
        help="Import one or more models into LM Studio.",
    )
    sync_parser.add_argument("models", nargs="*", help="Model names from `sdl scan`.")
    sync_parser.add_argument(
        "--all", action="store_true", help="Sync every discovered model."
    )
    sync_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the import command without changing state.",
    )
    sync_parser.add_argument(
        "--copy", action="store_true", help="Use LM Studio copy mode."
    )
    sync_parser.add_argument(
        "--hard-link", action="store_true", help="Use LM Studio hard-link mode."
    )
    sync_parser.add_argument(
        "--symbolic-link", action="store_true", help="Use LM Studio symbolic-link mode."
    )
    sync_parser.add_argument(
        "--direct",
        action="store_true",
        help="Import the Ollama blob directly (no staging alias).",
    )
    sync_parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    sync_parser.set_defaults(func=run_sync)

    status_parser = subparsers.add_parser(
        "status",
        parents=[common],
        help="Show discovered models and sync state.",
    )
    status_parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    status_parser.set_defaults(func=run_status)

    doctor_parser = subparsers.add_parser(
        "doctor",
        parents=[common],
        help="Check local StudioLink prerequisites.",
    )
    doctor_parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    doctor_parser.set_defaults(func=run_doctor)

    prune_parser = subparsers.add_parser(
        "prune",
        parents=[common],
        help="Remove staging aliases and state for models Ollama no longer has.",
    )
    prune_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be removed without deleting anything.",
    )
    prune_parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    prune_parser.set_defaults(func=run_prune)

    upgrade_parser = subparsers.add_parser(
        "upgrade",
        parents=[common],
        help="Check for and install newer StudioLink version.",
    )
    upgrade_parser.set_defaults(func=run_upgrade)

    help_parser = subparsers.add_parser(
        "help",
        parents=[common],
        help="Show the full command reference.",
    )
    help_parser.set_defaults(func=run_help)
    return parser


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    logger.setLevel(logging.DEBUG if verbose else logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not args_list:
        parser.print_help()
        return 0

    args = parser.parse_args(args_list)
    _configure_logging(getattr(args, "verbose", False))

    service = StudioLinkService()
    logger.debug("Configuration loaded:")
    logger.debug("  Ollama manifests: %s", service.config.ollama_manifests_dir)
    logger.debug("  Ollama blobs: %s", service.config.ollama_blobs_dir)
    logger.debug("  LM Studio models: %s", service.config.lmstudio_models_dir)
    logger.debug("  State file: %s", service.config.state_file)
    try:
        return int(args.func(args, service))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except LMStudioError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        if getattr(args, "verbose", False):
            logger.debug("Unhandled error:", exc_info=True)
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def run_scan(args: argparse.Namespace, service: StudioLinkService) -> int:
    models = service.scan()
    if args.json:
        print(json.dumps([_model_to_json(model) for model in models], indent=2))
        return 0

    if not models:
        print("No Ollama manifests were discovered.")
        return 0

    print(f"Discovered {len(models)} model(s):")
    for model in models:
        print(f"- {model.canonical_name} [{model.readiness.value}]")
        print(f"  blob: {model.blob_path or 'missing'}")
        for artifact in model.artifacts:
            if artifact.role.value != "model":
                print(
                    f"  {artifact.role.value}: "
                    f"{artifact.blob_path or 'missing'} "
                    f"[{artifact.readiness.value}]"
                )
        if model.issues:
            print(f"  issues: {'; '.join(model.issues)}")
    return 0


def run_sync(args: argparse.Namespace, service: StudioLinkService) -> int:
    mode = _resolve_link_mode(args)
    import_mode = ImportMode.DIRECT if args.direct else ImportMode.ALIAS
    results = service.sync(
        model_names=args.models,
        sync_all=bool(args.all),
        link_mode=mode,
        import_mode=import_mode,
        dry_run=bool(args.dry_run),
    )
    if args.json:
        print(json.dumps([_sync_result_to_json(item) for item in results], indent=2))
        return 0 if all(item.status != "error" for item in results) else 1

    for item in results:
        print(f"- {item.model.canonical_name}: {item.status}")
        print(f"  {item.message}")
    return 0 if all(item.status != "error" for item in results) else 1


def run_status(args: argparse.Namespace, service: StudioLinkService) -> int:
    entries = service.status()
    if args.json:
        print(json.dumps([_status_entry_to_json(entry) for entry in entries], indent=2))
        return 0

    if not entries:
        print("No Ollama manifests were discovered.")
        return 0

    synced = sum(1 for entry in entries if entry.synced)
    print(f"Discovered {len(entries)} model(s); {synced} tracked as synced.")
    for entry in entries:
        print(f"- {entry.model.canonical_name}: {entry.display_status}")
        if entry.sync_record is not None:
            imported_at = entry.sync_record.imported_at.astimezone(
                timezone.utc
            ).isoformat()
            print(f"  imported: {imported_at}")
        if entry.model.issues:
            print(f"  issues: {'; '.join(entry.model.issues)}")
    return 0


def run_doctor(args: argparse.Namespace, service: StudioLinkService) -> int:
    checks = service.doctor()
    if args.json:
        print(json.dumps([_doctor_check_to_json(check) for check in checks], indent=2))
        return 0 if all(check.ok for check in checks) else 1

    for check in checks:
        prefix = "OK" if check.ok else "FAIL"
        print(f"- {prefix}: {check.name} -> {check.details}")
    return 0 if all(check.ok for check in checks) else 1


def run_prune(args: argparse.Namespace, service: StudioLinkService) -> int:
    report = service.prune(dry_run=bool(args.dry_run))
    if args.json:
        print(json.dumps(report.to_json(), indent=2))
        return 0

    verb = "Would remove" if report.dry_run else "Removed"
    if not report.aliases and not report.records_removed:
        print("Nothing to prune.")
        return 0
    for item in report.aliases:
        print(f"- {verb} alias: {item.path} ({item.size} bytes) - {item.reason}")
    if report.records_removed:
        print(
            f"- {verb} {len(report.records_removed)} stale sync record(s): "
            + ", ".join(report.records_removed)
        )
    would_free = sum(item.size for item in report.aliases if item.would_free)
    if would_free:
        action = "would free" if report.dry_run else "freed"
        print(f"~{would_free} bytes {action} (aliases that were the last link).")
    return 0


def run_upgrade(args: argparse.Namespace, service: StudioLinkService) -> int:
    try:
        url = "https://pypi.org/pypi/studiolink/json"
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
            latest = str(data["info"]["version"])
    except Exception as exc:
        print(f"Error: Could not check for updates: {exc}", file=sys.stderr)
        return 1

    if _version_tuple(latest) <= _version_tuple(__version__):
        print(f"Already on latest version: {__version__}")
        return 0

    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "studiolink",
        "--upgrade",
        "--quiet",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        print(
            f"Error: Failed to upgrade (pip exited {result.returncode}):",
            file=sys.stderr,
        )
        for stream in (result.stdout, result.stderr):
            if stream and stream.strip():
                print(stream.strip(), file=sys.stderr)
        print(
            "Hint: on a system-managed Python (PEP 668), upgrade inside a "
            "virtualenv or via the tool you installed with (e.g. pipx).",
            file=sys.stderr,
        )
        return 1
    print(f"Upgraded from {__version__} to {latest}")
    return 0


def _version_tuple(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for piece in version.split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def run_help(args: argparse.Namespace, service: StudioLinkService) -> int:
    help_text = """StudioLink - Sync Ollama-downloaded GGUF models into LM Studio.

Available Commands:

  scan              Discover GGUF-backed Ollama models.
                    Usage: sdl scan [--json] [-v]

  sync              Import one or more models into LM Studio.
                    Usage: sdl sync <model> [<model>...] [--copy|--hard-link|--symbolic-link] [--direct] [--dry-run] [--json] [-v]
                           sdl sync --all [--copy|--hard-link|--symbolic-link] [--direct] [--dry-run] [--json] [-v]

  status            Show discovered models and sync state.
                    Usage: sdl status [--json] [-v]

  doctor            Check local StudioLink prerequisites.
                    Usage: sdl doctor [--json] [-v]

  prune             Safely remove unneeded staging aliases and stale state
                    after a complete Ollama library scan.
                    Usage: sdl prune [--dry-run] [--json] [-v]

  upgrade           Check for and install a newer StudioLink version.
                    Usage: sdl upgrade [-v]

  help              Show this help message.

Global Options:

  -v, --verbose     Enable verbose output (debug logging).
  --version         Show version information.
  --help            Show help for a specific command.

Examples:

  sdl scan                       # List all available models
  sdl -v scan                    # Scan with debug output
  sdl sync <modelname>           # Import a specific model
  sdl sync --all                 # Import all discovered models
  sdl status                     # Check sync status
  sdl doctor                     # Verify prerequisites
  sdl prune                      # Clean up orphaned import aliases

For more help on a specific command:
  sdl <command> --help
"""
    print(help_text)
    return 0


def _resolve_link_mode(args: argparse.Namespace) -> LinkMode | None:
    selected = [
        mode
        for flag, mode in (
            (args.copy, LinkMode.COPY),
            (args.hard_link, LinkMode.HARD_LINK),
            (args.symbolic_link, LinkMode.SYMBOLIC_LINK),
        )
        if flag
    ]
    if len(selected) > 1:
        raise ValueError("choose only one of --copy, --hard-link, or --symbolic-link")
    return selected[0] if selected else None


def _model_to_json(model: object) -> dict[str, object]:
    return {
        "canonical_name": getattr(model, "canonical_name"),
        "fully_qualified_name": getattr(model, "fully_qualified_name"),
        "blob_path": str(getattr(model, "blob_path"))
        if getattr(model, "blob_path")
        else None,
        "readiness": getattr(model, "readiness").value,
        "gguf_valid": getattr(model, "gguf_valid"),
        "issues": list(getattr(model, "issues")),
        "user_repo": getattr(model, "user_repo"),
        "artifacts": [
            {
                "role": artifact.role.value,
                "media_type": artifact.media_type,
                "digest": artifact.digest,
                "blob_path": (str(artifact.blob_path) if artifact.blob_path else None),
                "readiness": artifact.readiness.value,
                "gguf_valid": artifact.gguf_valid,
                "issues": list(artifact.issues),
            }
            for artifact in getattr(model, "artifacts", ())
        ],
    }


def _sync_result_to_json(result: SyncResult) -> dict[str, object]:
    return {
        "model": result.model.canonical_name,
        "status": result.status,
        "message": result.message,
        "record": None if result.record is None else result.record.to_json(),
    }


def _status_entry_to_json(entry: StatusEntry) -> dict[str, object]:
    return {
        "model": _model_to_json(entry.model),
        "synced": entry.synced,
        "status": entry.display_status,
        "sync_record": None
        if entry.sync_record is None
        else entry.sync_record.to_json(),
    }


def _doctor_check_to_json(check: DoctorCheck) -> dict[str, object]:
    return {
        "name": check.name,
        "ok": check.ok,
        "details": check.details,
    }
