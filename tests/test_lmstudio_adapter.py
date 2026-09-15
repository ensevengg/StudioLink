import json
import subprocess

import pytest

from studiolink.lmstudio_adapter import LMStudioAdapter, LMStudioError
from studiolink.models import LinkMode


def test_list_models_parses_json_inventory(make_config, monkeypatch):
    config = make_config()
    adapter = LMStudioAdapter(config)
    payload = [
        {
            "type": "llm",
            "modelKey": "mmproj-pair",
            "path": "ollama/llava/llava-deadbeef.gguf",
            "vision": True,
        }
    ]

    monkeypatch.setattr(
        adapter,
        "_run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, json.dumps(payload), ""
        ),
    )

    inventory = adapter.list_models()

    assert len(inventory) == 1
    assert inventory[0].model_key == "mmproj-pair"
    assert inventory[0].path == "ollama/llava/llava-deadbeef.gguf"
    assert inventory[0].vision is True


def test_list_models_rejects_invalid_json(make_config, monkeypatch):
    adapter = LMStudioAdapter(make_config())
    monkeypatch.setattr(
        adapter,
        "_run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "nope", ""),
    )

    with pytest.raises(LMStudioError, match="invalid model inventory JSON"):
        adapter.list_models()


def test_import_wraps_os_execution_error(make_config, monkeypatch):
    config = make_config()
    adapter = LMStudioAdapter(config)

    def missing_executable(*args, **kwargs):
        raise FileNotFoundError("lms is missing")

    monkeypatch.setattr(
        "studiolink.lmstudio_adapter.subprocess.run", missing_executable
    )

    with pytest.raises(LMStudioError, match="Could not execute LM Studio CLI"):
        adapter.import_model(
            "model.gguf",
            user_repo="ollama/model",
            link_mode=LinkMode.COPY,
        )
