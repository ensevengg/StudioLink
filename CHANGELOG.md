# Changelog

All notable changes to StudioLink are documented here. Versions follow
[Semantic Versioning](https://semver.org/).

## [0.3.0] - 2026-11-18

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
- Automatic link mode: without `--hard-link`, `--symbolic-link` or `--copy`,
  sync uses hard links when they work, otherwise symbolic links, otherwise
  copies, and says which it picked. Already-synced models keep their mode.
- Linux: finds Ollama's model store for the official installer, distro
  packages (`/var/lib/ollama` on Arch/CachyOS), a relocation set in the
  `ollama` systemd unit, and `~/.ollama/models`.
- `sdl doctor` reports the link mode sync will use, probing real links even
  before StudioLink's directories exist, and checks LM Studio model inventory
  (`lms ls --json`).

### Fixed

- Interrupted LM Studio imports are recovered on the next sync or prune.
- Sync verifies imported files and repairs missing ones instead of skipping.
- Sync and prune no longer run concurrently across processes.
- Identical model files shared by several Ollama tags are reused.
- Ollama blobs are validated and incomplete scans are reported.
- Symlinked imports and the hard-link fallback work on Windows.
- Symbolic-link and copy modes work when hard links are blocked by
  `fs.protected_hardlinks` or a read-only Ollama store (Linux system service).
- `--hard-link` failures explain why and suggest another mode.
- Prune removes the empty LM Studio folders an import leaves behind.
- Imports stay recognised when Linux renumbers devices across reboots (btrfs
  subvolumes, NVMe enumeration order).
- On Linux, a store that holds models wins over an empty one the Ollama
  service created, and `sdl prune` keeps imports whose weights still exist in
  another Ollama store instead of deleting them.
- Copy imports are no longer read back in full after copying; 0.2.0 copy
  records are compared once and then remembered.
- Leftover files from an interrupted import no longer abort sync or prune;
  `sdl doctor` lists them.
- `status` degrades gracefully when a library is unavailable; state writes are
  durable and tolerate a UTF-8 BOM.

## [0.2.0]

Initial release under the `ensevengg` organization.
