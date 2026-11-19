from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path

from studiolink.models import SyncRecord

logger = logging.getLogger("studiolink")


class StateStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        # Set when an unreadable state file could not be moved aside; saving
        # would then overwrite the only copy of the user's records.
        self._unrecoverable: str | None = None

    @contextmanager
    def lock(self):
        """Serialize import/prune transactions across processes; never wait forever."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with lock_path.open("a+b") as handle:
            if os.name == "nt":
                import msvcrt

                handle.seek(0, 2)
                if handle.tell() == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)

                def acquire():
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)

                def release():
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                def acquire():
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

                def release():
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

            try:
                acquire()
            except OSError as exc:
                raise RuntimeError(
                    "Another StudioLink sync/prune is running; retry when it finishes"
                ) from exc
            try:
                yield
            finally:
                release()

    def get_record(self, name: str) -> SyncRecord | None:
        """Get a single sync record by model name."""
        record_data = self._load_raw().get(name)
        return self._parse_record(name, record_data)

    def get_all_records(self) -> dict[str, SyncRecord]:
        """Get all sync records, skipping entries that fail to parse."""
        records: dict[str, SyncRecord] = {}
        for name, record_data in self._load_raw().items():
            record = self._parse_record(name, record_data)
            if record is not None:
                records[name] = record
        return records

    def upsert(self, record: SyncRecord) -> None:
        """Insert or update a sync record."""
        records = self._load_raw()
        records[record.canonical_name] = record.to_json()
        self._save_raw(records)

    def remove(self, name: str) -> None:
        """Delete a sync record by model name."""
        records = self._load_raw()
        if name in records:
            del records[name]
            self._save_raw(records)

    def save(self, records: dict[str, SyncRecord]) -> None:
        """Save all sync records at once (bulk operation)."""
        records_json = {name: record.to_json() for name, record in records.items()}
        self._save_raw(records_json)

    # Backwards-compatible alias.
    save_all = save

    @staticmethod
    def _parse_record(name: str, record_data: object) -> SyncRecord | None:
        if not isinstance(record_data, dict):
            return None
        try:
            return SyncRecord.from_json(record_data)
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Skipping corrupt sync record %r: %s", name, exc)
            return None

    def _save_raw(self, records: Mapping[str, object]) -> None:
        """Persist raw JSON dict to disk atomically and durably."""
        if self._unrecoverable is not None:
            raise RuntimeError(
                f"Refusing to overwrite the unreadable state file {self.path}: "
                f"{self._unrecoverable}. Move or repair it, then retry."
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"schema_version": 2, "sync_records": records}, indent=2)
        # A per-process temp name keeps concurrent writers from interleaving
        # writes into one file, and fsync before the replace keeps a crash
        # from leaving a truncated state file behind.
        tmp_path = self.path.parent / f"{self.path.name}.tmp-{os.getpid()}"
        try:
            with open(tmp_path, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    def _set_aside(self, problem: str) -> None:
        """Keep an unreadable state file instead of letting the next save erase it."""
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup = self.path.with_name(f"{self.path.name}.corrupt-{stamp}-{os.getpid()}")
        try:
            os.replace(self.path, backup)
        except FileNotFoundError:
            # Another process already moved it aside.
            return
        except OSError as exc:
            self._unrecoverable = f"{problem}; could not back it up: {exc}"
            logger.error(
                "Corrupted state file at %s: %s", self.path, self._unrecoverable
            )
            return
        logger.warning(
            "Corrupted state file at %s: %s. Saved it as %s and starting fresh; "
            "already-imported models will be re-imported.",
            self.path,
            problem,
            backup,
        )

    def _load_raw(self) -> dict[str, object]:
        """Load raw JSON dict from disk."""
        if not self.path.exists():
            self._unrecoverable = None
            return {}
        try:
            # utf-8-sig tolerates editors that save the file with a BOM;
            # a strict read would discard every record as "corrupted".
            payload = json.loads(self.path.read_text(encoding="utf-8-sig"))
        except (json.JSONDecodeError, UnicodeError) as exc:
            self._set_aside(str(exc))
            return {}
        if not isinstance(payload, dict):
            self._set_aside("expected a JSON object")
            return {}
        self._unrecoverable = None
        records = payload.get("sync_records", {})
        return records if isinstance(records, dict) else {}
