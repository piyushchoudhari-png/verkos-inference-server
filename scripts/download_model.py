#!/usr/bin/env python3
"""Download model weights to the host path declared in config.yaml.

Run this on the host before starting the container:
    python scripts/download_model.py --config /etc/inference-server/config.yaml

The script is idempotent: it skips download if the manifest exists and checksums pass.
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import yaml


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _write_manifest(weights_dir: Path, source: dict, files: list[Path]) -> None:
    checksums = {str(p.relative_to(weights_dir)): _sha256(p) for p in files}
    manifest = {"source": source, "checksums": checksums}
    (weights_dir / ".download_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Manifest written to {weights_dir / '.download_manifest.json'}")


def _verify_manifest(weights_dir: Path) -> bool:
    manifest_path = weights_dir / ".download_manifest.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text())
    for rel_path, expected in manifest["checksums"].items():
        full = weights_dir / rel_path
        if not full.exists():
            print(f"Missing file: {full}")
            return False
        if _sha256(full) != expected:
            print(f"Checksum mismatch: {full}")
            return False
    print("Manifest verified — weights are intact, skipping download.")
    return True


def _download_huggingface(repo_id: str, revision: str, dest: Path) -> list[Path]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface-hub not installed. Run: pip install huggingface-hub")
        sys.exit(1)

    print(f"Downloading {repo_id}@{revision} → {dest}")
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
    )
    return list(dest.rglob("*"))


def _download_s3(uri: str, dest: Path) -> list[Path]:
    try:
        import boto3
    except ImportError:
        print("ERROR: boto3 not installed. Run: pip install boto3")
        sys.exit(1)

    bucket, _, prefix = uri.removeprefix("s3://").partition("/")
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    downloaded: list[Path] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key.removeprefix(prefix).lstrip("/")
            local = dest / rel
            local.parent.mkdir(parents=True, exist_ok=True)
            print(f"  s3://{bucket}/{key} → {local}")
            s3.download_file(bucket, key, str(local))
            downloaded.append(local)
    return downloaded


def _copy_local(src: Path, dest: Path) -> list[Path]:
    shutil.copytree(src, dest, dirs_exist_ok=True)
    return list(dest.rglob("*"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: config not found at {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model = cfg.get("model", {})
    weights_dir = Path(model["path"])
    source = model.get("source")

    if not source:
        print("No model.source in config — nothing to download.")
        sys.exit(0)

    weights_dir.mkdir(parents=True, exist_ok=True)

    if _verify_manifest(weights_dir):
        sys.exit(0)

    source_type = source.get("type", "huggingface")

    if source_type == "huggingface":
        files = _download_huggingface(
            repo_id=source["repo_id"],
            revision=source.get("revision", "main"),
            dest=weights_dir,
        )
    elif source_type == "s3":
        files = _download_s3(uri=source["repo_id"], dest=weights_dir)
    elif source_type == "local-copy":
        files = _copy_local(src=Path(source["repo_id"]), dest=weights_dir)
    else:
        print(f"ERROR: unknown source type '{source_type}'")
        sys.exit(1)

    weight_files = [f for f in files if f.is_file() and f.name != ".download_manifest.json"]
    _write_manifest(weights_dir, source, weight_files)
    print("Done.")


if __name__ == "__main__":
    main()
