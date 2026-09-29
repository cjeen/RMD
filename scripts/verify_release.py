#!/usr/bin/env python3
"""Verify files in a prepared model release without network access."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(folder):
    folder = folder.resolve()
    manifest = json.loads((folder / "manifest.json").read_text())
    for entry in manifest["files"]:
        path = folder / entry["path"]
        if path.is_symlink() or not path.resolve().is_relative_to(folder):
            raise ValueError(f"Release must contain independent files: {path}")
        if path.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"Size mismatch: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {path}")
        print(f"OK {entry['path']}", flush=True)
    return len(manifest["files"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    print(f"Verified {verify(parser.parse_args().folder)} files.")
