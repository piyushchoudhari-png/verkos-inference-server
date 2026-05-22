#!/usr/bin/env python3
"""Generate, list, and revoke bearer API keys for the Verkos inference gateway.

Standalone CLI — stdlib only. Reads and writes ``keys.json`` (default
``./keys.json``), which the gateway loads at startup and reloads on SIGHUP.

The full key is printed to stdout exactly once at creation time. After that,
only the first 12 characters of any key are ever displayed.

Examples:
    python scripts/generate_api_key.py --name data-team
    python scripts/generate_api_key.py --name alice --keys-file /etc/verkos/keys.json
    python scripts/generate_api_key.py --list
    python scripts/generate_api_key.py --revoke data-team

The ``keys.json`` file is gitignored. In production, keep it in ``/etc/verkos/``
owned by the gateway process user with mode ``0600``.
"""

import argparse
import datetime as dt
import json
import os
import secrets
import sys
from pathlib import Path
from typing import Any, TypedDict

KEY_PREFIX = "sk-flytbase-airr-"
KEY_RANDOM_BYTES = 32
KEY_PREFIX_DISPLAY_CHARS = 12
DEFAULT_KEYS_FILE = Path("./keys.json")
KEYS_FILE_MODE = 0o600


class KeyRecord(TypedDict):
    """One entry in ``keys.json``."""

    name: str
    key: str
    created_at: str
    revoked_at: str | None


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with second precision."""
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _mint_key() -> str:
    """Generate a fresh ``sk-flytbase-airr-`` bearer key."""
    return f"{KEY_PREFIX}{secrets.token_urlsafe(KEY_RANDOM_BYTES)}"


def _load_keys(path: Path) -> list[KeyRecord]:
    """Load the keys file. Returns an empty list if the file does not exist."""
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return []
    data: Any = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON array of key records, got {type(data).__name__}")
    return data


def _save_keys(path: Path, records: list[KeyRecord]) -> None:
    """Atomically write ``records`` to ``path`` with mode 0600."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(records, indent=2) + "\n"
    # Create the file with restrictive permissions before writing the contents.
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, KEYS_FILE_MODE)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)
    # os.replace preserves the destination's mode if it already existed; force 0600 either way.
    os.chmod(path, KEYS_FILE_MODE)


def _find(records: list[KeyRecord], name: str) -> KeyRecord | None:
    """Return the first record with ``name``, or None."""
    return next((r for r in records if r["name"] == name), None)


def cmd_create(name: str, keys_file: Path) -> int:
    """Mint a new key for ``name`` and append it to ``keys_file``."""
    records = _load_keys(keys_file)
    if _find(records, name) is not None:
        print(
            f"error: a key named {name!r} already exists in {keys_file}. "
            "Names are append-only — choose a different name (e.g. add a version suffix).",
            file=sys.stderr,
        )
        return 2

    key = _mint_key()
    records.append(
        KeyRecord(name=name, key=key, created_at=_now_iso(), revoked_at=None),
    )
    _save_keys(keys_file, records)

    print(
        f"Key for {name!r} written to {keys_file}. The full key is printed once below and cannot be recovered later.",
        file=sys.stderr,
    )
    print(key)
    return 0


def cmd_list(keys_file: Path) -> int:
    """Print one row per record: name | created_at | revoked_at | key_prefix."""
    records = _load_keys(keys_file)
    if not records:
        print(f"(no keys in {keys_file})", file=sys.stderr)
        return 0
    name_w = max(len("name"), max(len(r["name"]) for r in records))
    header = f"{'name':<{name_w}}  {'created_at':<20}  {'revoked_at':<20}  key_prefix"
    print(header)
    print("-" * len(header))
    for r in records:
        prefix = r["key"][:KEY_PREFIX_DISPLAY_CHARS] + "..."
        revoked = r["revoked_at"] or "-"
        print(f"{r['name']:<{name_w}}  {r['created_at']:<20}  {revoked:<20}  {prefix}")
    return 0


def cmd_revoke(name: str, keys_file: Path) -> int:
    """Mark the record for ``name`` as revoked. Does not delete it."""
    records = _load_keys(keys_file)
    record = _find(records, name)
    if record is None:
        print(f"error: no key named {name!r} in {keys_file}", file=sys.stderr)
        return 2
    if record["revoked_at"] is not None:
        print(
            f"key {name!r} was already revoked at {record['revoked_at']}; nothing to do",
            file=sys.stderr,
        )
        return 0
    record["revoked_at"] = _now_iso()
    _save_keys(keys_file, records)
    print(f"revoked {name!r} at {record['revoked_at']}", file=sys.stderr)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="generate_api_key.py",
        description=(
            "Generate, list, or revoke bearer API keys for the Verkos inference gateway. "
            "Keys are stored in a JSON file (default ./keys.json) which is gitignored "
            "and should be kept with mode 0600."
        ),
    )
    p.add_argument(
        "--keys-file",
        type=Path,
        default=DEFAULT_KEYS_FILE,
        help=f"Path to the keys JSON file. Default: {DEFAULT_KEYS_FILE}",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--name", help="Mint a new key under this name.")
    g.add_argument("--list", dest="list_keys", action="store_true", help="List all records (no full keys shown).")
    g.add_argument("--revoke", metavar="NAME", help="Revoke the key under NAME (record is kept for audit).")
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = _build_parser().parse_args(argv)
    keys_file: Path = args.keys_file
    if args.list_keys:
        return cmd_list(keys_file)
    if args.revoke is not None:
        return cmd_revoke(args.revoke, keys_file)
    return cmd_create(args.name, keys_file)


if __name__ == "__main__":
    raise SystemExit(main())
