"""
Central registry of all onboarded companies, backed by a single JSON file
(data/registry.json).

Design decision — JSON, not SQLite: at the scale this platform targets
(tens of companies, not thousands), a single JSON file is simpler to
inspect, diff in git, and back up than a database file, and avoids adding
a SQLite dependency + migration story for what's fundamentally a small,
infrequently-written config table. If company count grows into the
thousands or writes become highly concurrent, SQLite (or a real DB) is the
right next step — the CompanyRegistry interface below is narrow enough
that swapping the storage backend wouldn't touch calling code.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class CompanyRecord(BaseModel):
    company_id: str
    name: str
    config_path: str
    data_path: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_ingestion: Optional[str] = None
    total_pdfs: int = 0
    total_chunks: int = 0


class Registry(BaseModel):
    version: int = 1
    companies: dict[str, CompanyRecord] = Field(default_factory=dict)


class CompanyAlreadyExistsError(Exception):
    pass


class CompanyNotFoundError(Exception):
    pass


class RegistryCorruptError(Exception):
    """Raised when registry.json can't be parsed. The corrupt file is backed
    up; callers must not silently proceed with an empty registry."""

    pass


class CompanyRegistry:
    """
    Args:
        registry_path: Path to the global registry.json (Settings.registry_path;
            this is NOT per-company — one file tracks every tenant).
    """

    def __init__(self, registry_path: Path):
        self.registry_path = Path(registry_path)

    def _load(self) -> Registry:
        if not self.registry_path.exists():
            return Registry()
        try:
            data = json.loads(self.registry_path.read_text(encoding="utf-8"))
            return Registry.model_validate(data)
        except (json.JSONDecodeError, ValueError) as exc:
            # H6 fix: a corrupt registry must NEVER be silently treated as
            # empty — the next write would then overwrite the file with only
            # the new data, permanently forgetting every registered company.
            # Back the corrupt file up and refuse to proceed until a human
            # looks at it.
            import time

            backup = self.registry_path.with_name(
                f"{self.registry_path.name}.corrupt.{int(time.time())}"
            )
            try:
                shutil.copy2(self.registry_path, backup)
            except OSError:
                pass
            raise RegistryCorruptError(
                f"registry.json is corrupt ({exc}); backed up to {backup}. "
                "Fix or remove the corrupt file before continuing."
            ) from exc

    def _save(self, registry: Registry) -> None:
        """Atomic write: .tmp then rename, so a crash mid-write never corrupts registry.json."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.registry_path.with_suffix(".json.tmp")
        tmp_path.write_text(registry.model_dump_json(indent=2), encoding="utf-8")
        tmp_path.replace(self.registry_path)

    def list_companies(self) -> list[CompanyRecord]:
        return list(self._load().companies.values())

    def get(self, company_id: str) -> CompanyRecord:
        registry = self._load()
        record = registry.companies.get(company_id)
        if record is None:
            raise CompanyNotFoundError(f"No company registered with id {company_id!r}")
        return record

    def exists(self, company_id: str) -> bool:
        return company_id in self._load().companies

    def add(self, company_id: str, name: str, config_path: str, data_path: str) -> CompanyRecord:
        """
        Args:
            company_id: Unique tenant identifier (must pass Settings' company_id validator).
            name: Human-readable company name.
            config_path: Path to this company's YAML config.
            data_path: Path to this company's isolated data directory.

        Raises:
            CompanyAlreadyExistsError: if company_id is already registered.
        """
        registry = self._load()
        if company_id in registry.companies:
            raise CompanyAlreadyExistsError(f"Company {company_id!r} is already registered")

        record = CompanyRecord(company_id=company_id, name=name, config_path=config_path, data_path=data_path)
        registry.companies[company_id] = record
        self._save(registry)
        return record

    def remove(self, company_id: str) -> None:
        """Unregister a company. Does NOT delete its data directory — that's a separate, explicit operation."""
        registry = self._load()
        if company_id not in registry.companies:
            raise CompanyNotFoundError(f"No company registered with id {company_id!r}")
        del registry.companies[company_id]
        self._save(registry)

    def update_stats(
        self,
        company_id: str,
        total_pdfs: Optional[int] = None,
        total_chunks: Optional[int] = None,
        mark_ingestion_now: bool = False,
    ) -> CompanyRecord:
        registry = self._load()
        record = registry.companies.get(company_id)
        if record is None:
            raise CompanyNotFoundError(f"No company registered with id {company_id!r}")

        if total_pdfs is not None:
            record.total_pdfs = total_pdfs
        if total_chunks is not None:
            record.total_chunks = total_chunks
        if mark_ingestion_now:
            record.last_ingestion = datetime.now(timezone.utc).isoformat()

        registry.companies[company_id] = record
        self._save(registry)
        return record
