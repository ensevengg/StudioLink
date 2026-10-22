# StudioLink

**Sync Ollama-downloaded GGUF models into LM Studio.**

StudioLink is a Python CLI tool that bridges Ollama and LM Studio. It treats Ollama as the source of truth, discovers GGUF-backed models from Ollama manifests, and imports them into LM Studio via the `lms` CLI—using hard links by default to save disk space.

## Features

- 🔍 **Scan** - Discover all GGUF models and supported vision projectors in your Ollama library
- 🔄 **Sync** - Import models into LM Studio with one command
- 📊 **Status** - Track which models are synced and their current state
- 🩺 **Doctor** - Verify prerequisites and diagnose issues
- 🧹 **Prune** - Safely remove orphaned aliases without breaking symbolic imports
- 🔗 **Smart linking** - Uses hard links by default; pass `--copy` when volumes differ
- 💾 **State tracking** - Remembers what's been synced to avoid re-importing

## Prerequisites

Before using StudioLink, ensure you have:

1. **Ollama** installed with a local model library (StudioLink detects current platform defaults and honors `OLLAMA_MODELS`)
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
- `--copy` - Use copy mode instead of hard links
- `--hard-link` - Force hard link mode (default)
- `--symbolic-link` - Use symbolic links
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
Remove aliases and sync state that are no longer needed, after first verifying that the Ollama library was scanned completely.

Space recovery depends on the import mode. A copied import can leave the staging alias as the last reference to Ollama's former blob, while an LM Studio hard link continues to retain the model data after the staging alias is removed. Aliases required by symbolic-link imports are preserved so LM Studio is not left with a broken link.

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
- Ollama manifests and blobs directories exist
- LM Studio models directory exists
- Hard-link filesystem/volume compatibility
- Hard-link permission on blobs (a real link is created and removed again, catching restrictions such as Linux `fs.protected_hardlinks`)
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
| Ollama models | `~\.ollama\models` | `/usr/share/ollama/.ollama/models` | `~/.ollama/models` | `STUDIOLINK_OLLAMA_MODELS_DIR` (falls back to Ollama's own `OLLAMA_MODELS`) |
| LM Studio models | LM Studio's configured `downloadsFolder` | LM Studio's configured `downloadsFolder` | LM Studio's configured `downloadsFolder` | `STUDIOLINK_LMSTUDIO_MODELS_DIR` |
| State directory | `~\.studiolink` | `~/.studiolink` | `~/.studiolink` | `STUDIOLINK_STATE_DIR` |

## How It Works

1. **Scanning** - Reads manifest files from the configured Ollama models directory
2. **Artifact Resolution** - Locates the primary model and optional projector GGUF in Ollama's content-addressed blob directory
3. **Validation** - Verifies every required artifact has a valid digest, exists, and starts with GGUF magic bytes
4. **Import Aliases** - Creates digest-keyed, human-readable `.gguf` aliases; projectors use the LM Studio-recognized `mmproj-` prefix. Copy mode uses temporary aliases on Ollama's filesystem so the state and LM Studio directories may live on other volumes
5. **LM Studio Import** - Uses one `lms import` call per artifact, placing each model/projector pair in the same digest-isolated repository so a re-pull cannot leave LM Studio choosing an older projector
6. **State Tracking** - Saves resumable per-artifact sync records to schema-v2 `~/.studiolink/state.json`; schema-v1 records remain readable
7. **Reconciliation** - Status and sync verify every expected target and confirm projector bundles through `lms ls --json`
8. **Pruning** - Pruning is blocked after unavailable or incomplete Ollama scans and preserves all aliases required by symbolic or partial imports

## Troubleshooting

### "model blob is missing from the Ollama blob store"
The model manifest exists but the actual blob file has been deleted or purged by Ollama. Re-download it:
```powershell
ollama pull <model-name>
```

### Hard link fails (cross-volume)
Hard-link imports require the Ollama blob store, StudioLink alias directory, and LM Studio models directory to live on the same filesystem/volume. If they do not, retry with copy mode; copy mode creates its temporary naming alias on Ollama's filesystem and can copy to LM Studio on another volume:

```powershell
sdl sync <model-name> --copy
```

The StudioLink state directory does not need to share Ollama's volume when using copy mode.

### Deleted a model in Ollama but disk space wasn't freed
Run `sdl prune` to remove aliases that are no longer needed. If LM Studio imported the model with `--hard-link`, its final hard link intentionally continues to retain the data; removing only the staging alias cannot free those bytes. Symbolic-link aliases are preserved while LM Studio depends on them.

### LM Studio import fails
Run `sdl doctor` to verify:
- LM Studio is installed
- `lms` CLI is in PATH
- Import directories are writable

## Design Notes

- **Ollama as source of truth** - StudioLink never changes Ollama manifests or content-addressed blobs; copy mode may briefly create a `.studiolink-imports` alias directory on the same filesystem
- **Hard links by default** - Space-efficient, creates references rather than copies
- **No silent fallbacks** - If a hard link is impossible (cross-volume), the sync fails loudly; use `--copy` explicitly
- **State persistence** - Tracks imports to avoid redundant operations; state is written atomically and corrupt records are skipped, never crash
- **Safe pruning** - Orphaned aliases are cleaned up only after a complete source scan; symbolic-link dependencies are retained
- **Supported bundles** - Any number of Ollama models can be synced; each manifest may contain one primary model GGUF and at most one projector GGUF
- **Explicit limitations** - Manifests with multiple primary models/projectors or Ollama adapter, draft, or tensor artifacts are reported as unsupported rather than partially imported
- **CLI-only** - No GUI, designed for automation and scripting

## License

MIT License - See [LICENSE](LICENSE) for details.
