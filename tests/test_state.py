import json

import pytest
from conftest import DIGEST_A, make_sync_record

from studiolink.models import SyncRecord
from studiolink.state import StateStore


def write_raw_state(path, payload, *, raw=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(raw if raw is not None else json.dumps(payload), encoding="utf-8")


class TestRoundtrip:
    def test_missing_file_returns_empty(self, tmp_path):
        store = StateStore(tmp_path / "state.json")
        assert store.get_all_records() == {}
        assert store.get_record("llama3:1b") is None

    def test_upsert_and_get(self, tmp_path):
        store = StateStore(tmp_path / "state.json")
        record = make_sync_record("llama3:1b", DIGEST_A)
        store.upsert(record)

        restored = store.get_record("llama3:1b")
        assert restored == record
        assert store.get_all_records() == {"llama3:1b": record}

    def test_bulk_save_and_remove(self, tmp_path):
        path = tmp_path / "state.json"
        store = StateStore(path)
        first = make_sync_record("a:1", DIGEST_A)
        second = make_sync_record("b:1", DIGEST_A)
        store.save({"a:1": first, "b:1": second})
        assert set(store.get_all_records()) == {"a:1", "b:1"}

        store.remove("a:1")
        assert set(store.get_all_records()) == {"b:1"}

    def test_save_all_alias(self, tmp_path):
        store = StateStore(tmp_path / "state.json")
        store.save_all({"a:1": make_sync_record("a:1", DIGEST_A)})
        assert store.get_record("a:1") is not None


class TestCorruptionTolerance:
    def test_corrupt_json_starts_fresh(self, tmp_path):
        path = tmp_path / "state.json"
        write_raw_state(path, None, raw="not json at all")
        assert StateStore(path).get_all_records() == {}

    def test_non_dict_payload_starts_fresh(self, tmp_path):
        path = tmp_path / "state.json"
        write_raw_state(path, None, raw="[1, 2, 3]")
        assert StateStore(path).get_all_records() == {}

    def test_unknown_link_mode_record_is_skipped(self, tmp_path):
        # Regression: one bad record used to raise ValueError and break
        # status/sync for every model.
        path = tmp_path / "state.json"
        good = make_sync_record("good:1", DIGEST_A).to_json()
        bad = make_sync_record("bad:1", DIGEST_A).to_json()
        bad["link_mode"] = "nonsense-mode"
        write_raw_state(
            path, {"schema_version": 1, "sync_records": {"bad:1": bad, "good:1": good}}
        )

        store = StateStore(path)
        records = store.get_all_records()
        assert set(records) == {"good:1"}
        assert store.get_record("bad:1") is None

    def test_non_dict_record_is_skipped(self, tmp_path):
        path = tmp_path / "state.json"
        good = make_sync_record("good:1", DIGEST_A).to_json()
        write_raw_state(
            path, {"schema_version": 1, "sync_records": {"bad:1": 42, "good:1": good}}
        )
        assert set(StateStore(path).get_all_records()) == {"good:1"}

    def test_missing_sync_records_key(self, tmp_path):
        path = tmp_path / "state.json"
        write_raw_state(path, {"schema_version": 1})
        assert StateStore(path).get_all_records() == {}


class TestAtomicWrites:
    def test_no_temp_file_left_behind(self, tmp_path):
        path = tmp_path / "state.json"
        store = StateStore(path)
        store.save({"a:1": make_sync_record("a:1", DIGEST_A)})

        assert path.exists()
        assert not (tmp_path / "state.json.tmp").exists()
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == 1
        assert "a:1" in payload["sync_records"]

    def test_overwrite_keeps_consistent_state(self, tmp_path):
        path = tmp_path / "state.json"
        store = StateStore(path)
        store.upsert(make_sync_record("a:1", DIGEST_A))
        store.upsert(make_sync_record("b:1", DIGEST_A))
        assert set(store.get_all_records()) == {"a:1", "b:1"}


class TestRecordSchema:
    def test_record_serializes_link_mode(self, tmp_path):
        path = tmp_path / "state.json"
        store = StateStore(path)
        store.upsert(make_sync_record("m:1", DIGEST_A))
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["sync_records"]["m:1"]["link_mode"] == "hard-link"

    def test_record_from_json_requires_fields(self):
        with pytest.raises(KeyError):
            SyncRecord.from_json({"canonical_name": "m:1"})
