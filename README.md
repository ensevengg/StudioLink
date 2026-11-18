# StudioLink

**Sync Ollama-downloaded GGUF models into LM Studio.**

StudioLink is a Python CLI tool that bridges Ollama and LM Studio. It treats Ollama as the source of truth, discovers GGUF-backed models from Ollama manifests, and imports them into LM Studio via the `lms` CLI—using hard links when the filesystem allows them, and symbolic links otherwise, to save disk space.

## Features

- 🔍 **Scan** - Discover all GGUF models and supported vision projectors in your Ollama library
- 🔄 **Sync** - Import models into LM Studio with one command
- 📊 **Status** - Track which models are synced and their current state
- 🩺 **Doctor** - Verify prerequisites and diagnose issues
- 🧹 **Prune** - Remove obsolete managed imports and aliases to release pinned disk space
- 🔗 **Smart linking** - Uses hard links when possible and switches to symbolic links (or copies) automatically when they aren't, e.g. across volumes or btrfs subvolumes
- 💾 **State tracking** - Remembers what's been synced to avoid re-importing

## Prerequisites

Before using StudioLink, ensure you have:

1. **Ollama** installed with a local model library (StudioLink detects platform defaults, including Linux system-service and distro-package stores, and honors `OLLAMA_MODELS`)
2. **LM Studio** with the `lms` CLI installed
3. **Python 3.12+**

Default executable locations are detected per platform (e.g. `ollama.exe` under `AppData` on Windows; `ollama` from your `PATH` and `~/.lmstudio/bin/lms` on Linux/macOS). Override with environment variables if yours differ - see [Configuration](#configuration).

## Installation

```powershell
pip install studiolink
```

To upgrade to the latest version:

```powershell
sdl upgrade
```

## Quick Start

```powershell
# Scan for available models
sdl scan

# Import a specific model
sdl sync <modelname>

# Import all discovered models
sdl sync --all

# Check sync status
sdl status

# Verify everything is set up correctly
sdl doctor
```

## Commands

### `sdl scan`
Discover all GGUF-backed Ollama models available for import.

```powershell
# Basic scan
sdl scan

# Output as JSON
sdl scan --json

# Verbose output (debug logging)
sdl -v scan
```

**Output:**
```
Discovered 3 model(s):
- <modelname> [ready]
  blob: C:\Users\...\ollama\models\blobs\sha256-...
- <modelname2> [stale]
  blob: C:\Users\...\ollama\models\blobs\sha256-...
  issues: model blob is missing from the Ollama blob store
```

### `sdl sync`
Import one or more models into LM Studio.

```powershell
# Import specific model(s)
sdl sync <modelname>
sdl sync <model1> <model2> <model3>

# Import all discovered models
sdl sync --all

# Preview without making changes (dry run)
sdl sync <modelname> --dry-run

# Use a specific link mode
sdl sync <modelname> --hard-link
sdl sync <modelname> --copy
sdl sync <modelname> --symbolic-link

# Import the opaque Ollama blob path directly (advanced)
sdl sync <modelname> --direct

# Verbose output
sdl -v sync --all
```

**Options:**
- `--all` - Sync every discovered model
- `--dry-run` - Preview the import without making changes
- `--copy` - Copy the weights (uses disk space, but keeps working if Ollama deletes the model)
- `--hard-link` - Force hard links; fails if they are not possible
- `--symbolic-link` - Use symbolic links

Without a mode flag, sync probes the machine and uses hard links if they work, otherwise symbolic links, otherwise copies, and prints which mode it picked. Models that are already synced keep the mode they were imported with.
- `--direct` - Import a single-artifact Ollama blob directly without a human-readable alias; model-plus-projector bundles require aliases

### `sdl status`
Show discovered models and their sync state.

```powershell
sdl status
sdl status --json
sdl -v status
```

**Output:**
```
Discovered 3 model(s); 1 tracked as synced.
- <modelname>: synced
  imported: 2026-04-12T10:30:00+00:00
- <modelname2>: stale
  issues: model blob is missing from the Ollama blob store
- <modelname3>: pending
```

### `sdl prune`
Remove tracked LM Studio imports, staging aliases, and sync state for models that Ollama no longer has or has replaced.

Hard-linked weights remain on disk until **all** links are removed, including the LM Studio destination. `sdl prune` removes both the tracked destination and its staging alias. It preserves files whose identity has changed and refuses cleanup after an incomplete source scan or when either model library is unavailable. It also recovers abandoned import transactions and temporary copy aliases. Files already present in LM Studio without a verifiable StudioLink record are left alone.

```powershell
# Preview what would be removed
sdl prune --dry-run

# Actually remove orphaned aliases and stale sync records
sdl prune

# Output as JSON
sdl prune --json
```

### `sdl doctor`
Check local StudioLink prerequisites and diagnose issues.

```powershell
sdl doctor
sdl doctor --json
```

**Checks include:**
- Ollama executable exists
- LM Studio CLI (`lms`) exists
- Ollama manifests (`manifests-v2/` from Ollama 0.40+, or legacy `manifests/`) and blobs directories exist
- LM Studio models directory exists
- Default link mode: which mode sync will use without a flag, and why. A real link is created and removed again (even before StudioLink's directories exist), catching different volumes, btrfs subvolumes, and Linux `fs.protected_hardlinks`
- LM Studio model inventory (`lms ls --json`; required to verify projector bundles)
- Discovered Ollama models
- Blob presence validation
- GGUF header validation

### `sdl help`
Show all available commands and their descriptions.

```powershell
sdl help
```

## Global Options

All commands support these global options:

- `-v, --verbose` - Enable verbose output (debug logging)
- `--version` - Show version information
- `--help` - Show help for the specific command

```powershell
# Examples
sdl -v scan              # Scan with debug logging
sdl sync --help          # Show help for sync command
sdl --version            # Show version
```

## Model Readiness States

When scanning or checking status, models can have these states:

| State | Description | Action Required |
|-------|-------------|-----------------|
| **ready** | Every required model/projector blob exists and has a valid GGUF header | Ready to sync |
| **stale** | A required model/projector blob is missing | Run `ollama pull <model>` to restore |
| **invalid** | An artifact is malformed or the manifest uses an unsupported artifact combination | Inspect the reported issues |
| **synced** | Already imported into LM Studio | No action needed |
| **pending** | Ready but not yet synced | Run `sdl sync` |

## Configuration

Default paths are platform-aware; every one can be overridden with an environment variable:

| Setting | Windows default | Linux default | macOS default | Environment Variable |
|---------|-----------------|---------------|---------------|---------------------|
| Ollama executable | `ollama` from `PATH`, then the standard app path | `ollama` from `PATH` (fallback `/usr/bin/ollama`) | `ollama` from `PATH` (fallback `/usr/bin/ollama`) | `STUDIOLINK_OLLAMA_EXE` |
| LM Studio CLI | `lms` from `PATH`, then `~\.lmstudio\bin\lms.exe` | `lms` from `PATH` (fallback `~/.lmstudio/bin/lms`) | `lms` from `PATH` (fallback `~/.lmstudio/bin/lms`) | `STUDIOLINK_LMS_EXE` |
| Ollama models | `~\.ollama\models` | First existing of: the `ollama` systemd unit's `OLLAMA_MODELS`, `/usr/share/ollama/.ollama/models` (official installer), `/var/lib/ollama` (Arch/CachyOS package), `~/.ollama/models` | `~/.ollama/models` | `STUDIOLINK_OLLAMA_MODELS_DIR` (falls back to Ollama's own `OLLAMA_MODELS`) |
| LM Studio models | LM Studio's configured `downloadsFolder` | LM Studio's configured `downloadsFolder` | LM Studio's configured `downloadsFolder` | `STUDIOLINK_LMSTUDIO_MODELS_DIR` |
| State directory | `~\.studiolink` | `~/.studiolink` | `~/.studiolink` | `STUDIOLINK_STATE_DIR` |

The LM Studio models override must match the models directory selected in LM Studio; it does not change where `lms import` writes files. A mismatch is reported before importing.

## How It Works

1. **Scanning** - Reads named manifests from both `manifests-v2/` (Ollama 0.40+, links to content-addressed manifest blobs) and the legacy `manifests/` tree, preferring v2 when a name exists in both. For Ollama manifest lists it picks the llama.cpp GGUF child, falling back to the `ggml` child; internal `llamacpp:<digest>` downgrade tags are ignored
2. **Artifact Resolution** - Locates the primary model and optional projector GGUF in Ollama's content-addressed blob directory
3. **Validation** - Checks every required artifact's SHA-256 digest syntax, GGUF magic bytes, and the blob size declared in the manifest
4. **Import Aliases** - Creates digest-keyed, human-readable `.gguf` aliases; projectors use the LM Studio-recognized `mmproj-` prefix. Copy mode uses temporary aliases on Ollama's filesystem, removed after the import; other modes use `~/.studiolink/imports/`
5. **LM Studio Import** - Uses one `lms import` call per artifact to stage and verify the requested link/copy, then publishes each destination without overwriting an existing file. Each model/projector pair lands in the same digest-isolated repository so a re-pull cannot leave LM Studio choosing an older projector. Failed imports clean up partial files; subsequent sync or prune recovers abandoned transactions after a process crash
6. **State Tracking** - Saves resumable per-artifact destination paths and file identities to schema-v2 `~/.studiolink/state.json` (written atomically; schema-v1 records remain readable). Sync and prune operations share a process lock.
7. **Reconciliation** - Status and sync verify every imported file before skipping, repair missing destinations, and confirm projector bundles through `lms ls --json`. An import whose Ollama blob was deleted stays synced only while it is the unchanged file StudioLink recorded.
8. **Cleanup** - Successful replacement imports remove the previous managed destinations and aliases; a failed re-pull keeps the previous import tracked. `sdl prune` removes obsolete tracked imports and orphaned aliases after a complete Ollama scan; its dry run counts shared file data once.

## Troubleshooting

### "model blob is missing from the Ollama blob store"
The model manifest exists but the actual blob file has been deleted or purged by Ollama. Re-download it:
```powershell
ollama pull <model-name>
```

### "Ollama MLX (safetensors) models are not supported"
The model only has an MLX build in Ollama (common on Apple Silicon), which has no GGUF file for LM Studio to import. Pull a GGUF variant of the model, or download it in LM Studio directly.

### Hard link fails
Hard links need the Ollama blob store, the StudioLink state directory, and the LM Studio models directory on the same filesystem - and on Linux, separate btrfs subvolumes (such as CachyOS's default `@` and `@home`) count as different filesystems. When Ollama runs as a system service, its blobs belong to the `ollama` user and `fs.protected_hardlinks` also blocks linking them.

Without a mode flag, sync detects this and uses symbolic links instead; `sdl doctor` shows the mode it will pick. If you passed `--hard-link`, drop it or choose another mode:

```sh
sdl sync <model-name> --symbolic-link   # no extra disk space; breaks if Ollama deletes the model
sdl sync <model-name> --copy            # full copy; independent of Ollama
```

Copy mode creates its temporary naming alias on Ollama's filesystem when it can, and otherwise uses a symbolic link in StudioLink's own directory. Symbolic-link mode requires permission to create symlinks on Windows (Developer Mode).

### Deleted a model in Ollama but disk space wasn't freed
Both LM Studio imports and StudioLink staging aliases can keep a deleted Ollama blob alive. Run `sdl prune --dry-run` to preview cleanup, then `sdl prune` to remove the obsolete managed files. Untracked LM Studio files may still hold links and need separate inspection.

### LM Studio import fails
Run `sdl doctor` to verify:
- LM Studio is installed
- `lms` CLI is in PATH
- Import directories are writable

## Design Notes

- **Ollama as source of truth** - StudioLink never changes Ollama manifests or content-addressed blobs; copy mode may briefly create a `.studiolink-imports` alias directory on the same filesystem
- **Hard links when possible** - Space-efficient, creates references rather than copies
- **No silent fallbacks** - An explicit mode flag never falls back; it fails with a hint. Without a flag, sync prints the mode it picked and why
- **State persistence** - Tracks and verifies actual imports to avoid redundant operations; state is written atomically and corrupt records are skipped. Verified existing files can restore lost state.
- **Copy deduplication** - Tags with identical weights share one destination copy through hard links when the destination filesystem supports them.
- **Dry runs** - Preview sync actions without replacing aliases or invoking an import.
- **Prune over leak** - Obsolete tracked destinations and orphaned aliases are cleaned up with `sdl prune` instead of pinning disk space forever; symbolic-link dependencies are retained
- **Supported bundles** - Any number of Ollama models can be synced; each manifest may contain one primary model GGUF and at most one projector GGUF
- **Explicit limitations** - Manifests with multiple primary models/projectors or Ollama adapter, draft, or tensor artifacts are reported as unsupported rather than partially imported
- **CLI-only** - No GUI, designed for automation and scripting

## License

MIT License - See [LICENSE](LICENSE) for details.
