# Import and storage validation — 2026-09-30

Validation was completed before committing on `fix/import-lifecycle-validation`,
based on `f9df9a3`. Local `main` remained at `221229d` throughout validation.

## Focused commit validation

The original combined commit (`ecb7fba`) is preserved locally at
`backup/import-lifecycle-before-split`. It was replaced on the fix branch with
focused commits. Every implementation commit includes its regression tests;
the final test commit adds the real-CLI and mixed-operation suites.

Before each commit, the complete suite present at that point passed on all three
Python versions, along with Ruff lint/format checks, Pyright, a source distribution
and wheel build, and `git diff --check`.

| Commit | Scope | Python 3.12 | Python 3.13 | Python 3.14 |
| --- | --- | --- | --- | --- |
| `64dbaa9` | validate Ollama blobs and surface incomplete scans | 139 passed | 139 passed | 139 passed |
| `b4a193d` | serialize sync and prune across processes | 142 passed | 142 passed | 142 passed |
| `966d460` | recover interrupted LM Studio imports | 155 passed | 155 passed | 155 passed |
| `d7e5f67` | verify and repair imported model files | 182 passed | 182 passed | 182 passed |
| `fb4df6f` | reclaim obsolete imports without breaking dependent links | 199 passed | 199 passed | 199 passed |
| `83164b7` | reuse identical model copies across tags | 201 passed | 201 passed | 201 passed |
| `f85820e` | exercise real lms imports and mixed lifecycle operations | 238 passed | 211 passed, 27 skipped | 211 passed, 27 skipped |

The documentation commit repeats the final checks. The rebuilt `src/` and
`tests/` trees match the original combined commit exactly; documentation adds
this validation record. Native-platform limits below remain unchanged.

## Results

| Check | Result |
| --- | --- |
| Linux, Python 3.12.14, including real `lms` | 238 passed |
| Linux, Python 3.13.15 | 211 passed, 27 opt-in real-CLI tests skipped |
| Linux, Python 3.14.7 | 211 passed, 27 opt-in real-CLI tests skipped |
| Ruff lint and formatting | Passed |
| Pyright | 0 errors, 0 warnings |
| Source distribution and wheel build | Passed |
| Wheel installed in a fresh Python 3.12 environment outside the checkout | Passed |
| Installed `sdl` version/help, scan, status, dry-run sync | Passed |
| Git whitespace/conflict checks | Passed |

The live CLI was `lms` commit `69d945a`. Its tests run in a Linux bubblewrap
mount namespace with the host filesystem read-only, a temporary writable fixture
library, an isolated LM Studio settings directory, and networking disabled.
The user's model libraries are not modified. The 27 cases cover the three link
modes, destination repair, lost state, repulls, pruning, all nine mode transitions
with and without state loss, identical-tag copy deduplication, and direct imports.

## Failure cases exercised

| Scenario | Method and assertion |
| --- | --- |
| Partial import, disk-full error, timeout, interruption | Injected subprocess failures after writing a real partial file; partial data and staging aliases disappear, retry succeeds |
| Process dies during a transaction | A child writes a partial import and calls `os._exit`; recovery removes its abandoned files |
| Concurrent operations and dead process locks | A real child holds the OS lock; another operation is refused; killing the child releases the lock |
| Full disk during state persistence | Injected `ENOSPC`; previous JSON survives atomic replacement failure; an existing completed copy is adopted without another import |
| Cross-filesystem imports | Real paths on separate filesystems (`tmp_path` and `/dev/shm`); hard-link mode fails without silently copying, copy and symbolic-link modes work |
| Unsupported destination hard links | Injected unsupported-link error exercises the actual Linux no-overwrite rename fallback |
| Missing or changed destination, source replacement, moved library | Filesystem tests verify status detects the change and sync repairs or safely refuses it |
| Unavailable library or incomplete source scan | Pruning refuses to delete imports, aliases, or state |
| Foreign replacement or untracked symbolic dependency | User files and aliases required by other imports remain intact |
| Prune permission failure | The failed item retains its record; a subsequent prune can finish |
| Identical tags, including mode conversion | Destination inode identity proves one independent copy is shared instead of duplicating weights |
| Shared direct-import destination | Both tags remain healthy after switching the shared target to copy mode |
| Removed or superseded models | Managed destination and alias are removed; shared-inode accounting avoids counting the same data twice |
| Dry run | Existing imported files and aliases are not replaced and no import is invoked |
| Mixed lifecycle operations | Ten deterministic seeds, 30 actions each: repulls, target deletion, state loss, dry runs, and mode changes, followed by complete tracked cleanup |

Focused regression tests were observed failing before the fixes for lost-state
mode conversion, duplicate copies during conversion, and shared direct targets.
The mixed-operation tests also exposed an unrealistic fixture rewrite of an
existing content-addressed blob; the fixture now reuses that blob as Ollama does.

## Behavior changed

- Sync and status verify the actual destination, requested link mode, and source
  relationship instead of trusting a saved digest alone.
- Imports stage in a recoverable transaction and publish only after verification.
  Failed imports and abandoned transactions no longer leave permanent partial
  models. Mode conversion replaces the managed file atomically.
- Repulls and pruning remove verified obsolete destinations as well as aliases,
  so an overlooked hard link cannot indefinitely pin obsolete weights.
- Copy imports reuse verified identical weights across tags, including when
  converting existing links to copies. Temporary copy aliases are cleaned up.
- OS locks serialize sync/prune operations sharing a state file. Lost state can
  be recovered from a verified existing destination without another full copy.
- Malformed manifests, incomplete scans, missing mounts, mismatched LM Studio
  paths, and replaced files receive more defensive handling.

## Reproduction

From the checkout, with `uv` available:

```bash
STUDIOLINK_TEST_REAL_LMS=1 uv run --no-project --python 3.12 --with pytest python -m pytest -q
uv run --no-project --python 3.13 --with pytest python -m pytest -q
uv run --no-project --python 3.14 --with pytest python -m pytest -q
uv run --no-project --python 3.12 --with ruff ruff check src tests
uv run --no-project --python 3.12 --with ruff ruff format --check src tests
uv run --no-project --python 3.12 --with pyright pyright
uv run --no-project --python 3.12 --with build python -m build
git diff --check
```

The real-CLI suite requires Linux, working bubblewrap user/mount namespaces, and
`lms` on PATH. It explicitly skips when prerequisites or its opt-in flag are absent;
check the skip count before treating a run as live-CLI validation. Package checks
built both distributions, installed the wheel into a temporary virtual environment,
verified imports came from site-packages, and ran the installed `sdl` entry point
outside the checkout. Temporary distributions and environments were removed.

## Remaining validation limits

- Native Windows and macOS were not available. Windows locking, symlink privileges,
  and platform-specific rename behavior still need native runs. Existing CI defines
  Linux/Windows Python 3.12–3.14 jobs; those remote jobs have not been run for this
  changes.
- Fixtures contain tiny GGUF headers and payloads. These checks validate importing
  and file lifecycle, not model loading/inference or multi-gigabyte throughput.
- Disk exhaustion, timeouts, and permission failures are injected. The tests do not
  fill a physical SSD, cut power, or kill the real CLI during a large copy.
- Actual exFAT, NTFS, network shares, and filesystem durability after power loss
  were not tested. Inode/link-count assertions establish sharing and logical
  reclamation, not a hardware SSD space measurement.
- Files with lost ownership evidence are deliberately retained when their content
  or dependencies cannot be verified; arbitrary pre-existing duplicates are not
  automatically deleted.
