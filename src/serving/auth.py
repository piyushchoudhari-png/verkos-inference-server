"""Bearer-key auth backed by ``keys.json``.

See architecture doc §A.5 and §C.3. The gateway reads ``keys.json`` at startup
and on SIGHUP so that ``scripts/generate_api_key.py --revoke`` takes effect
without a restart. A missing file is fatal at startup.
"""

import json
import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class KeyStore:
    """In-memory cache of non-revoked bearer keys.

    Reads are lock-free against an immutable snapshot dict; reloads swap the
    snapshot atomically under a lock. Designed for hot lookups on every
    request with rare reloads.

    Attributes:
        path: The keys file the store mirrors.
    """

    def __init__(self, path: Path) -> None:
        """Initialize the store but do not load yet — call :meth:`reload` first."""
        self.path = path
        self._lock = threading.Lock()
        self._key_to_name: dict[str, str] = {}

    def reload(self) -> int:
        """Re-read ``keys.json`` and swap the in-memory map. Returns active key count.

        Raises:
            FileNotFoundError: The keys file does not exist.
            ValueError: The keys file is malformed.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"keys file not found: {self.path}")
        raw = self.path.read_text(encoding="utf-8")
        records: Any = json.loads(raw) if raw.strip() else []
        if not isinstance(records, list):
            raise ValueError(f"{self.path}: expected a JSON array, got {type(records).__name__}")

        snapshot: dict[str, str] = {}
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"{self.path}: key record must be a dict, got {type(record).__name__}")
            if record.get("revoked_at") is not None:
                continue
            key = record.get("key")
            name = record.get("name")
            if not isinstance(key, str) or not isinstance(name, str):
                raise ValueError(f"{self.path}: each record needs str 'key' and 'name'")
            snapshot[key] = name

        with self._lock:
            self._key_to_name = snapshot
        logger.info("keys reloaded", extra={"active_keys": len(snapshot), "path": str(self.path)})
        return len(snapshot)

    def lookup(self, key: str) -> str | None:
        """Return the client name for ``key`` or None if unknown/revoked."""
        return self._key_to_name.get(key)


def parse_bearer(authorization_header: str | None) -> str | None:
    """Extract the bearer token from an ``Authorization`` header.

    Returns None if the header is missing, malformed, or does not use the
    ``Bearer`` scheme. The token itself is not validated here — the caller
    runs it through :meth:`KeyStore.lookup`.
    """
    if not authorization_header:
        return None
    parts = authorization_header.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    token = parts[1].strip()
    return token or None
