import pytest

from studiolink.lmstudio_adapter import LMStudioAdapter, LMStudioError
from studiolink.models import LinkMode


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
