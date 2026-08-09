from __future__ import annotations

import dataclasses
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from configs.constants import APP_NAME, VERSION_NUMBER
from services.annotation_stats import AnnotationStats


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _atomic_write_text(path: Path, text: str) -> None:
    """Write via a temp file + os.replace so a crash mid-write can never
    truncate review.json. os.replace is atomic on Windows and POSIX, so a
    reader sees either the old file or the fully-written new one — never a
    half-written one. The cumulative store is long-lived, so this protects
    months of accumulated review history from a single bad write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class ReviewStore:
    """In-memory + on-disk store of per-file review comments and annotation
    stats, keyed by absolute annotation file path. Backs review.json."""

    def __init__(self, entries: dict | None = None):
        self.entries: dict[str, dict] = entries or {}

    @classmethod
    def load(cls, path: Path) -> "ReviewStore":
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    entries = data.get("entries", {})
                    if isinstance(entries, dict):
                        return cls(entries=entries)
        except Exception:
            pass
        return cls()

    def save(self, path: Path, *, meta_extra: dict | None = None) -> None:
        payload = {
            "meta": {
                "app_name": APP_NAME,
                "version": VERSION_NUMBER,
                "generated_at": _now_iso(),
                "total_entries": len(self.entries),
                **(meta_extra or {}),
            },
            "entries": self.entries,
        }
        _atomic_write_text(path, json.dumps(payload, indent=2))

    def set_comment(self, key: str, text: str, *, filename: str = "", annotation_path: str = "") -> None:
        entry = self.entries.setdefault(key, {
            "filename": filename,
            "annotation_path": annotation_path or key,
        })
        entry["comment"] = text
        entry["comment_updated_at"] = _now_iso()

    def get_comment(self, key: str) -> str:
        return self.entries.get(key, {}).get("comment", "")

    def upsert_stats(self, key: str, stats: AnnotationStats) -> None:
        entry = self.entries.setdefault(key, {})
        entry.update(dataclasses.asdict(stats))

    def rekey(self, old_key: str, new_key: str, *, filename: str | None = None) -> bool:
        """Move an entry's key when its file is moved/renamed so the review
        comment and stats follow the file instead of becoming an orphan.
        No-op (returns False) if there's no entry under old_key or the key
        is unchanged."""
        if old_key == new_key or old_key not in self.entries:
            return False
        entry = self.entries.pop(old_key)
        entry["annotation_path"] = new_key
        if filename:
            entry["filename"] = filename
        # If something was already recorded under the destination path,
        # prefer the entry that actually carries the review history.
        self.entries[new_key] = entry
        return True
