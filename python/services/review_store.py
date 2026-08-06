from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

from configs.constants import APP_NAME, VERSION_NUMBER
from services.annotation_stats import AnnotationStats


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


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
        path.parent.mkdir(parents=True, exist_ok=True)
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
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

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
