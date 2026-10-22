# Branch reliability audit

**Date:** 2026-09-02  
**Scope:** `main` (`221229d`), `stabilize-sync-lifecycle` (`3d5c60b`), and
`support-artifact-bundles` (`1090dbc`)  
**Method:** read-only `git show`/`git diff` inspection, the existing worktrees,
repository tests, small temporary-directory probes, and primary platform sources.

## Executive conclusion

The proposed four fixes are well chosen, but the bundles branch has two additional
merge blockers:

- Re-pulling a vision bundle when only one artifact digest changes reuses the
  unchanged artifact from the old digest-keyed LM Studio directory and imports
  the changed artifact into the new directory. The split can either fail forever
  or, worse, be falsely confirmed against the old bundle.
- A valid manifest that lists the projector before the model imports the
  projector first, then tries to persist a projector-only partial record.
  `primary_artifact` raises `StopIteration`, aborting the entire batch.

Recommended pre-merge order:

1. **P1:** normalize bundle imports to model-first and never serialize a partial
   `SyncRecord` without exactly one model artifact.
2. **P1:** keep every artifact of a changed vision bundle co-located; do not reuse
   an unchanged artifact from a different `import_user_repo` without importing or
   linking it into the new repository.
3. **P1:** translate stale-alias `unlink()` and staging `mkdir()` failures to the
   per-model `RuntimeError` result, with two-model continuation tests.
4. **P2:** make bundles `status()` degrade to pending when `lms ls --json` fails.
5. **P2:** read state with `utf-8-sig`, with a real-record regression test.
6. **P2/platform:** add an actual hard-link permission probe and actionable Linux
   error; `st_dev` equality alone is insufficient.
7. **P2/conditional correctness:** make `same_file()` return false when either
   inode/file identifier is zero. A size comparison is only an extra rejection
   check, not proof that equal-sized files are the same file.

The full existing suites pass (main 117, stabilize 135, bundles 156), so the
listed failures are coverage gaps rather than known failing tests.

## Branch relationship

`support-artifact-bundles` is not an alternative to `stabilize-sync-lifecycle`:
its merge base with stabilize is exactly `3d5c60b`, so bundles contains all of
stabilize plus `7bbf601` and `1090dbc`. Review and merge planning should treat
the work as one stack.

## Verified bugs

| Claim | Verdict and evidence |
|---|---|
| Batch abort on alias filesystem errors | **Confirmed on all three refs.** The stale `unlink()` and `mkdir()` occur before the `try` that wraps only `os.link()`: [main 198-204](https://github.com/ensevengg/StudioLink/blob/221229ddd9df6176134a1da695fd82c0a21499cb/src/studiolink/syncer.py#L198-L204), [stabilize 231-237](https://github.com/ensevengg/StudioLink/blob/3d5c60b1bc6b3927b1fb2be196d0fd87c591526c/src/studiolink/syncer.py#L231-L237), [bundles 457-463](https://github.com/ensevengg/StudioLink/blob/1090dbcb80436e3a0a4be4d7bc3db95b8b853acf/src/studiolink/syncer.py#L457-L463). Callers catch only `RuntimeError` ([bundles 194-207](https://github.com/ensevengg/StudioLink/blob/1090dbcb80436e3a0a4be4d7bc3db95b8b853acf/src/studiolink/syncer.py#L194-L207)), while the outer batch loop has no catch ([74-80](https://github.com/ensevengg/StudioLink/blob/1090dbcb80436e3a0a4be4d7bc3db95b8b853acf/src/studiolink/syncer.py#L74-L80)). A raw `PermissionError`/`FileExistsError` therefore aborts the batch. Existing isolation tests cover only LM Studio failures ([tests 436-469](https://github.com/ensevengg/StudioLink/blob/1090dbcb80436e3a0a4be4d7bc3db95b8b853acf/tests/test_syncer.py#L436-L469)). |
| Bundles `status` crashes when `lms` is broken | **Confirmed, bundles only.** Status conditionally calls `list_models()` with no guard ([77-99](https://github.com/ensevengg/StudioLink/blob/1090dbcb80436e3a0a4be4d7bc3db95b8b853acf/src/studiolink/service.py#L77-L99)); the adapter translates execution and JSON failures to `LMStudioError` ([66-110](https://github.com/ensevengg/StudioLink/blob/1090dbcb80436e3a0a4be4d7bc3db95b8b853acf/src/studiolink/lmstudio_adapter.py#L66-L110), [149-176](https://github.com/ensevengg/StudioLink/blob/1090dbcb80436e3a0a4be4d7bc3db95b8b853acf/src/studiolink/lmstudio_adapter.py#L149-L176)). `inventory=None` is already interpreted as unsynced/pending for projector records ([service 392-397](https://github.com/ensevengg/StudioLink/blob/1090dbcb80436e3a0a4be4d7bc3db95b8b853acf/src/studiolink/service.py#L392-L397)), so the proposed fallback fits the current model. Current status tests cover an empty inventory, not an inventory exception ([tests 430-461](https://github.com/ensevengg/StudioLink/blob/1090dbcb80436e3a0a4be4d7bc3db95b8b853acf/tests/test_service.py#L430-L461)). |
| UTF-8 BOM drops state | **Confirmed on all three refs.** `_load_raw()` uses `encoding="utf-8"` and turns the resulting JSON decode error into `{}`/“Starting fresh”: [main 72-82](https://github.com/ensevengg/StudioLink/blob/221229ddd9df6176134a1da695fd82c0a21499cb/src/studiolink/state.py#L72-L82), [bundles 71-81](https://github.com/ensevengg/StudioLink/blob/1090dbcb80436e3a0a4be4d7bc3db95b8b853acf/src/studiolink/state.py#L71-L81). Sync then sees no records and re-imports. `utf-8-sig` on reads is the narrow correct fix; writes can remain UTF-8. No BOM regression test exists. |

## Additional bundles merge blockers

### Projector-first manifests abort the batch

The adapter preserves the manifest's model/projector layer order
([adapter 133-162](https://github.com/ensevengg/StudioLink/blob/1090dbcb80436e3a0a4be4d7bc3db95b8b853acf/src/studiolink/ollama_adapter.py#L133-L162)),
and sync imports in that order. After *each* imported artifact it creates and
saves a partial record ([syncer 245-265](https://github.com/ensevengg/StudioLink/blob/1090dbcb80436e3a0a4be4d7bc3db95b8b853acf/src/studiolink/syncer.py#L245-L265)).
But serialization unconditionally resolves `primary_artifact`
([models 211-217](https://github.com/ensevengg/StudioLink/blob/1090dbcb80436e3a0a4be4d7bc3db95b8b853acf/src/studiolink/models.py#L211-L217),
[240-248](https://github.com/ensevengg/StudioLink/blob/1090dbcb80436e3a0a4be4d7bc3db95b8b853acf/src/studiolink/models.py#L240-L248)).
A projector-only partial has no primary, so `next(...)` raises `StopIteration`;
the outer batch loop does not catch it.

A temporary probe with layers `[projector C, model A]` produced artifact order
`['projector', 'model']`, imported only the projector, then raised
`StopIteration`. Normalize supported artifacts to model-first before import (or
defer partial persistence until a primary exists) and add both a standalone and
two-model batch regression. The current fixture's model-first order does not
establish a general ordering invariant.

### Changed vision bundles are split across old and new repositories

The bundles branch keys `import_user_repo` by *all* artifact digest prefixes
([models 120-129](https://github.com/ensevengg/StudioLink/blob/1090dbcb80436e3a0a4be4d7bc3db95b8b853acf/src/studiolink/models.py#L120-L129)).
On re-sync it reuses an artifact solely by `(role, digest)` and target existence
([syncer 146-188](https://github.com/ensevengg/StudioLink/blob/1090dbcb80436e3a0a4be4d7bc3db95b8b853acf/src/studiolink/syncer.py#L146-L188)),
but imports a changed artifact into the newly computed repository
([190-217](https://github.com/ensevengg/StudioLink/blob/1090dbcb80436e3a0a4be4d7bc3db95b8b853acf/src/studiolink/syncer.py#L190-L217)).
LM Studio pairing requires both files in one directory
([the branch's end-to-end validation](https://github.com/ensevengg/StudioLink/blob/1090dbcb80436e3a0a4be4d7bc3db95b8b853acf/docs/research/multi-artifact-validation.md#L8-L13)),
and reconciliation tests the new primary path for vision
([374-397](https://github.com/ensevengg/StudioLink/blob/1090dbcb80436e3a0a4be4d7bc3db95b8b853acf/src/studiolink/syncer.py#L374-L397)).

A temporary-directory probe synced `(model=A, projector=C)`, changed only the
model to `B`, then synced again. The first sync was `synced`; the second imported
one file and returned `error: LM Studio did not index the projector bundle as
vision-capable`. State pointed to:

```text
.../llava--bundle-bbbbbbbbbbbb-cccccccccccc/<model>
.../llava--bundle-aaaaaaaaaaaa-cccccccccccc/<projector>
```

Retry reuses both split targets and repeats the same error. The inverse change is
more dangerous: with `(model=A, projector=C)` already installed, changing only
the projector to `D` reuses the old primary in the old `A-C` directory (where old
projector C remains), imports D alone into the new `A-D` directory, then confirms
vision against the *old* A-C inventory entry. State and `status` can therefore
say synced while LM Studio still pairs A with stale projector C.

The fix should make reuse conditional on the target's parent being the current
destination, or re-import/re-link every required artifact into the new bundle
directory. Add tests for changing only the primary digest and only the projector
digest, asserting co-location and the exact inventory path used for confirmation.

## Conditional platform risks

### Linux protected hard links

**Strongly substantiated; the kernel mechanism was reproduced in WSL, but not
against an installed Ollama service.** With `fs.protected_hardlinks=1`, a
hard-link attempt by an unprivileged user to a root-owned `0644` source was
denied, while the same probe succeeded after the source was changed to `0666`.
That isolates the expected ownership/write-permission mechanism; the probe host
did not have a real Ollama installation. Stabilize changes the Linux default
store to `/usr/share/ollama/.ollama/models`
([config 18-21](https://github.com/ensevengg/StudioLink/blob/3d5c60b1bc6b3927b1fb2be196d0fd87c591526c/src/studiolink/config.py#L18-L21)).
Ollama's official system-service instructions create and run as the dedicated
`ollama` user ([official Linux docs](https://github.com/ollama/ollama/blob/205a0426905d45f5ed0d120f3c671e3573e01b41/docs/linux.mdx#adding-ollama-as-a-startup-service-recommended)),
and Ollama explicitly chmods newly created blobs to `0644`
([official source](https://github.com/ollama/ollama/blob/205a0426905d45f5ed0d120f3c671e3573e01b41/manifest/layer.go#L52-L59)).
With `fs.protected_hardlinks=1`, Linux rejects links by users who neither own nor
have read/write access to the source
([kernel documentation](https://docs.kernel.org/admin-guide/sysctl/fs.html#protected-hardlinks)).

StudioLink always first calls `os.link(blob, alias)`, including copy mode
([stabilize 208-238](https://github.com/ensevengg/StudioLink/blob/3d5c60b1bc6b3927b1fb2be196d0fd87c591526c/src/studiolink/syncer.py#L208-L238)).
The doctor check compares only `st_dev` values
([service 323-348](https://github.com/ensevengg/StudioLink/blob/3d5c60b1bc6b3927b1fb2be196d0fd87c591526c/src/studiolink/service.py#L323-L348)),
so a typical single-filesystem machine can pass while link creation returns
`EPERM`; copy staging can also fail earlier if the user cannot create
`.studiolink-imports` under the Ollama-owned store. An actual reversible
`os.link()` probe using a real blob and each relevant destination is materially
better than a permission-bit or volume-only check. The error should explain
ownership/`protected_hardlinks` and the supported remediation; a doctor warning
alone does not make the default hard-link workflow work.

### Zero/unknown inode identifiers

**The code defect is confirmed; the exact exFAT/FAT incidence was not reproduced.**
Every ref treats equal `(st_dev, st_ino)` as conclusive
([bundles 34-43](https://github.com/ensevengg/StudioLink/blob/1090dbcb80436e3a0a4be4d7bc3db95b8b853acf/src/studiolink/syncer.py#L34-L43)).
Python only promises uniqueness when `st_ino` is nonzero
([official `stat_result` docs](https://docs.python.org/3/library/os.html#os.stat_result)).
Thus two distinct files reported as `(same device, 0)` are misclassified and a
wrong pre-existing alias can be reused. Return false if either identifier is
zero. Comparing `st_size` is useful as another mismatch check, but equal sizes
do not establish identity and therefore do not fully fix stale-weight reuse.

## Design-level reliability gaps

| Claim | Verdict and refinement |
|---|---|
| No fsync before replace | **Confirmed on all refs.** State is written with `Path.write_text()` and immediately replaced, with no file flush/fsync and no parent-directory fsync ([bundles 63-69](https://github.com/ensevengg/StudioLink/blob/1090dbcb80436e3a0a4be4d7bc3db95b8b853acf/src/studiolink/state.py#L63-L69)). Atomic namespace replacement is not a power-loss durability guarantee. A robust sequence is unique temp file -> flush/fsync file -> replace -> fsync parent where supported. |
| No locking/fixed temp | **Confirmed on all refs.** Every writer uses `state.json.tmp` ([bundles 67-69](https://github.com/ensevengg/StudioLink/blob/1090dbcb80436e3a0a4be4d7bc3db95b8b853acf/src/studiolink/state.py#L67-L69)) and read/modify/write operations have no inter-process lock. Concurrent commands can lose updates; the shared temp name also permits replace races and `FileNotFoundError`. Use a state-scoped inter-process lock around the full transaction and unique same-directory temp files. |
| Prune ignores copy staging | **Confirmed on both improvement refs.** Copy aliases live under `<ollama_models>/.studiolink-imports` ([bundles 421-433](https://github.com/ensevengg/StudioLink/blob/1090dbcb80436e3a0a4be4d7bc3db95b8b853acf/src/studiolink/syncer.py#L421-L433)), while prune scans only `config.import_staging_dir` ([service 176-213](https://github.com/ensevengg/StudioLink/blob/1090dbcb80436e3a0a4be4d7bc3db95b8b853acf/src/studiolink/service.py#L176-L213)). A process crash or cleanup failure can therefore leave a hard link that pins bytes indefinitely. |
| Dry-run has filesystem side effects | **Refine.** It really does create a staging directory and hard link before invoking `lms --dry-run`; it is not intrinsically read-only ([bundles 194-227](https://github.com/ensevengg/StudioLink/blob/1090dbcb80436e3a0a4be4d7bc3db95b8b853acf/src/studiolink/syncer.py#L194-L227)). However, successful stabilize/bundles runs unlink the alias *and rmdir the staging leaf* ([399-419](https://github.com/ensevengg/StudioLink/blob/1090dbcb80436e3a0a4be4d7bc3db95b8b853acf/src/studiolink/syncer.py#L399-L419)); because a real blob implies `ollama_models_dir` already exists, successful copy dry-runs do not normally leave parent directories. Persistent empty `.studiolink-imports` is an error/crash path (notably `mkdir` succeeds then `os.link` fails), not the normal success path. Main does leave its state staging directory after a successful dry-run because it unlinks only the file ([main 151-159](https://github.com/ensevengg/StudioLink/blob/221229ddd9df6176134a1da695fd82c0a21499cb/src/studiolink/syncer.py#L151-L159)). |
| Missing source blob: text skipped, vision errors | **Confirmed in bundles.** A synced text model returns before the stale check, while projector models deliberately bypass that return and reach the stale error ([syncer 104-126](https://github.com/ensevengg/StudioLink/blob/1090dbcb80436e3a0a4be4d7bc3db95b8b853acf/src/studiolink/syncer.py#L104-L126)). Choose and test one policy; reporting stale consistently is least surprising. |
| Old LM Studio imports are orphaned | **Confirmed.** Filenames include digest prefixes ([models 139-149](https://github.com/ensevengg/StudioLink/blob/1090dbcb80436e3a0a4be4d7bc3db95b8b853acf/src/studiolink/models.py#L139-L149)); vision repositories include all digest prefixes ([120-129](https://github.com/ensevengg/StudioLink/blob/1090dbcb80436e3a0a4be4d7bc3db95b8b853acf/src/studiolink/models.py#L120-L129)). Prune removes staging aliases and state only, never LM Studio targets. Text re-pulls leave old digest files; vision re-pulls leave old bundle directories. Deletion needs an explicit ownership/safety policy because those directories are user-visible LM Studio data. |
| Upgrade prerelease comparison | **Confirmed, unchanged.** `_version_tuple()` strips non-digits per dotted component, so `0.2.1rc1` becomes `(0, 2, 11)` ([CLI 320-325](https://github.com/ensevengg/StudioLink/blob/221229ddd9df6176134a1da695fd82c0a21499cb/src/studiolink/cli.py#L320-L325)); the test codifies the faulty behavior ([tests 240-241](https://github.com/ensevengg/StudioLink/blob/221229ddd9df6176134a1da695fd82c0a21499cb/tests/test_cli.py#L240-L241)). Use standards-aware version parsing. |
| Pip upgrade unpinned | **Confirmed, unchanged.** The command installs unpinned `studiolink --upgrade` after separately reading a reported version ([CLI 272-301](https://github.com/ensevengg/StudioLink/blob/221229ddd9df6176134a1da695fd82c0a21499cb/src/studiolink/cli.py#L272-L301)). Pin the fetched version if the success message is meant to be authoritative. |
| ANSI/control-sequence spoofing | **Confirmed, print-only, unchanged.** Canonical names derived from manifest path components are printed directly by scan/sync/status ([CLI 183-188](https://github.com/ensevengg/StudioLink/blob/221229ddd9df6176134a1da695fd82c0a21499cb/src/studiolink/cli.py#L183-L188), [206-208](https://github.com/ensevengg/StudioLink/blob/221229ddd9df6176134a1da695fd82c0a21499cb/src/studiolink/cli.py#L206-L208), [223-232](https://github.com/ensevengg/StudioLink/blob/221229ddd9df6176134a1da695fd82c0a21499cb/src/studiolink/cli.py#L223-L232)). JSON output remains machine-safe encoding; sanitize controls in terminal rendering. |

## “Probed safe” claims

| Claim | Verdict and scope |
|---|---|
| Prune is symlink-safe | **Destructive action is safe, wording needs nuance.** `is_file()`, `same_file()`, and `stat()` follow a planted symlink for metadata, but deletion is `path.unlink()`, which removes the link entry rather than its target ([bundles service 179-204](https://github.com/ensevengg/StudioLink/blob/1090dbcb80436e3a0a4be4d7bc3db95b8b853acf/src/studiolink/service.py#L179-L204)). So “target never deleted” is accurate; “never followed” is not. An expected-name symlink to the current blob can also satisfy `same_file()` and be preserved rather than unlinked. |
| `--direct` never deletes the Ollama blob | **Confirmed.** Direct mode uses the blob as `alias_path` ([bundles 190-192](https://github.com/ensevengg/StudioLink/blob/1090dbcb80436e3a0a4be4d7bc3db95b8b853acf/src/studiolink/syncer.py#L190-L192)), and cleanup explicitly excludes direct mode ([399-413](https://github.com/ensevengg/StudioLink/blob/1090dbcb80436e3a0a4be4d7bc3db95b8b853acf/src/studiolink/syncer.py#L399-L413)). |
| Ctrl-C cleans temporary aliases | **Confirmed for ordinary Python interruption.** Import is inside `try/finally`, so `KeyboardInterrupt` runs cleanup before propagating ([bundles 209-227](https://github.com/ensevengg/StudioLink/blob/1090dbcb80436e3a0a4be4d7bc3db95b8b853acf/src/studiolink/syncer.py#L209-L227)). This cannot cover hard process termination or power loss. |
| Concurrent alias-creation race is non-corrupting | **Confirmed for the absent-alias race.** One `os.link` wins; the other receives an `OSError` translated to `RuntimeError` ([bundles 459-464](https://github.com/ensevengg/StudioLink/blob/1090dbcb80436e3a0a4be4d7bc3db95b8b853acf/src/studiolink/syncer.py#L459-L464)). It fails one model rather than overwriting content. This does not make state writes concurrency-safe. |
| Artifact ordering is consistent | **Refuted as a safety claim.** The adapter and syncer consistently preserve *manifest* order ([adapter 133-162](https://github.com/ensevengg/StudioLink/blob/1090dbcb80436e3a0a4be4d7bc3db95b8b853acf/src/studiolink/ollama_adapter.py#L133-L162), [syncer 158-188](https://github.com/ensevengg/StudioLink/blob/1090dbcb80436e3a0a4be4d7bc3db95b8b853acf/src/studiolink/syncer.py#L158-L188)), but they do not enforce model-first. A projector-first manifest crashes while saving the first partial record, as detailed above. Normalize semantic order rather than relying on the current fixture. |
| Schema v2 downgrades cleanly to 0.2.0 | **No-crash compatibility confirmed, but it is lossy.** Bundles keeps legacy primary fields at the record top level ([models 240-260](https://github.com/ensevengg/StudioLink/blob/1090dbcb80436e3a0a4be4d7bc3db95b8b853acf/src/studiolink/models.py#L240-L260)), and 0.2.0 ignores extra fields while parsing ([main models 117-129](https://github.com/ensevengg/StudioLink/blob/221229ddd9df6176134a1da695fd82c0a21499cb/src/studiolink/models.py#L117-L129)). But the next 0.2.0 bulk `save()` serializes legacy records and schema 1, discarding projector/vision metadata ([main state 44-70](https://github.com/ensevengg/StudioLink/blob/221229ddd9df6176134a1da695fd82c0a21499cb/src/studiolink/state.py#L44-L70)). Safe means “does not crash,” not round-trip preservation. |
| Corrupt manifest JSON protects aliases | **Confirmed on stabilize and bundles, not main.** Invalid JSON becomes an `INVALID` model rather than disappearing ([bundles adapter 116-126](https://github.com/ensevengg/StudioLink/blob/1090dbcb80436e3a0a4be4d7bc3db95b8b853acf/src/studiolink/ollama_adapter.py#L116-L126)); prune protects aliases for every unready model with a record ([service 163-183](https://github.com/ensevengg/StudioLink/blob/1090dbcb80436e3a0a4be4d7bc3db95b8b853acf/src/studiolink/service.py#L163-L183)). Main has no `unready_names` protection and can remove that alias. |

## Test additions worth requiring with the fixes

- Two-model batches where the first alias hits `PermissionError` from `mkdir()`
  and from stale-alias `unlink()`; the second model must still sync.
- Vision `status()` where `list_models()` raises `LMStudioError`; output must be
  pending and the command successful.
- A BOM-prefixed state file containing a real record, not just empty JSON.
- Projector-first valid manifests, including a two-model batch; the model must be
  imported first or no invalid projector-only partial may be saved.
- Vision re-pulls changing only the model digest and only the projector digest;
  all imported paths must share the current bundle parent and reconciliation
  must succeed.
- Mocked zero inode/file identifiers for two distinct files; `same_file()` must
  return false even when their sizes match.
- Copy dry-run/link-failure cleanup and prune coverage for
  `.studiolink-imports` leftovers.
