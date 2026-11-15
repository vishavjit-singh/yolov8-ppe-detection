"""Download and unpack the Ultralytics Construction-PPE dataset (CC BY 4.0), with a pinned checksum."""
from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip"
SHA256 = "e5817aa9c73f33e525d76ca6bc6af4e4eac9dcea9bcfef1cfd1a71c09fbc1232"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_extract(zip_path: Path, dest: Path) -> None:
    """Extract, refusing entries that would land outside ``dest`` (zip-slip)."""
    dest = dest.resolve()
    with zipfile.ZipFile(zip_path) as z:
        for member in z.infolist():
            target = (dest / member.filename).resolve()
            if dest not in target.parents and target != dest:
                raise ValueError(f"unsafe path in archive: {member.filename}")
        z.extractall(dest)


def download(dest_dir: Path, url: str = URL, expected_sha256: str | None = SHA256,
             zip_path: Path | None = None) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / "construction-ppe"
    if (target / "data.yaml").exists():
        return target
    zip_path = zip_path or dest_dir / "construction-ppe.zip"
    if not zip_path.exists():
        tmp = zip_path.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=120) as r, tmp.open("wb") as f:
            shutil.copyfileobj(r, f)
        tmp.rename(zip_path)
    if expected_sha256 and sha256(zip_path) != expected_sha256:
        raise ValueError(f"checksum mismatch for {zip_path}")
    safe_extract(zip_path, dest_dir)
    return target
