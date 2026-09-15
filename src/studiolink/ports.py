from __future__ import annotations

from typing import Protocol, runtime_checkable

from studiolink.models import ImportResult, LinkMode, OllamaModel, OllamaScanReport


@runtime_checkable
class OllamaPort(Protocol):
    def scan_models(self) -> list[OllamaModel]: ...

    def scan_report(self) -> OllamaScanReport: ...


@runtime_checkable
class LMStudioPort(Protocol):
    def get_version(self) -> str | None: ...

    def get_import_capabilities(self) -> set[LinkMode]: ...

    def import_model(
        self,
        source_path: str,
        *,
        user_repo: str,
        link_mode: LinkMode,
        dry_run: bool = False,
    ) -> ImportResult: ...
