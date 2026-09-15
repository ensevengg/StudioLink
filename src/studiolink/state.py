from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path

from studiolink.models import SyncRecord

logger = logging.getLogger("studiolink")


class StateStore:
    def __init__(self, path: Path) -> None:
        self.path = path

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
        """Persist raw JSON dict to disk atomically."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"schema_version": 1, "sync_records": records}, indent=2)
        tmp_path = self.path.parent / (self.path.name + ".tmp")
        tmp_path.write_text(payload, encoding="utf-8")
        os.replace(tmp_path, self.path)

    def _load_raw(self) -> dict[str, object]:
        """Load raw JSON dict from disk."""
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning(
                "Corrupted state file at %s: %s. Starting fresh.", self.path, exc
            )
            return {}
        if not isinstance(payload, dict):
            logger.warning(
                "Corrupted state file at %s: expected a JSON object. Starting fresh.",
                self.path,
            )
            return {}
        records = payload.get("sync_records", {})
        return records if isinstance(records, dict) else {}
