# Changelog

All notable changes to StudioLink are documented here. Versions follow
[Semantic Versioning](https://semver.org/).

## [0.3.0] - Unreleased

### ⚠️ Behavior changes — read before upgrading

- **`sdl prune` now deletes LM Studio files.** In 0.2.0 it only removed
  staging aliases and sync records. It now also removes the LM Studio model
  files that StudioLink imported, when Ollama no longer has the model or has
  replaced it. Files StudioLink has no verifiable record of are left alone, and
  prune refuses to run after an incomplete Ollama scan or when either model
  library is unavailable. Run `sdl prune --dry-run` first to preview.
- **State file upgraded to schema v2.** `~/.studiolink/state.json` is migrated
  automatically, and 0.2.0 records remain readable. Downgrading to 0.2.0 will
  not crash, but the next 0.2.0 save drops vision/projector metadata.

### Added

- Support for Ollama 0.40+ model storage (`manifests-v2/`), alongside the
  legacy `manifests/` tree. Ollama manifest lists use their GGUF child;
  MLX-only models are reported as unsupported.
- Sync of vision models that ship with a projector (`mmproj`) file; both files
  land together in LM Studio.
- `sdl sync <model> --direct` to import a single-file model without a
  human-readable alias.
- `sdl doctor` checks hard-link permission and LM Studio model inventory
  (`lms ls --json`).

### Fixed

- Interrupted LM Studio imports are recovered on the next sync or prune.
- Sync verifies imported files and repairs missing ones instead of skipping.
- Sync and prune no longer run concurrently across processes.
- Identical model files shared by several Ollama tags are reused.
- Ollama blobs are validated and incomplete scans are reported.
- Symlinked imports and the hard-link fallback work on Windows.
- `status` degrades gracefully when a library is unavailable; state writes are
  durable and tolerate a UTF-8 BOM.

## [0.2.0]

Initial release under the `ensevengg` organization.
