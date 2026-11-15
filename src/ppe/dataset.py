"""YOLO-format dataset tooling: load data.yaml, validate labels, compute statistics, check splits.

Works on any Ultralytics-style detection dataset (``images/<split>`` + ``labels/<split>`` with one
``class cx cy w h`` row per box, coordinates normalised to [0, 1]).
"""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass(frozen=True)
class Label:
    cls: int
    cx: float
    cy: float
    w: float
    h: float

    def xyxy(self, width: int, height: int) -> tuple[float, float, float, float]:
        return ((self.cx - self.w / 2) * width, (self.cy - self.h / 2) * height,
                (self.cx + self.w / 2) * width, (self.cy + self.h / 2) * height)


@dataclass
class LabelIssue:
    file: str
    line: int
    problem: str


@dataclass
class DatasetSpec:
    root: Path
    splits: dict[str, str]
    names: dict[int, str]

    @property
    def class_ids(self) -> dict[str, int]:
        return {v: k for k, v in self.names.items()}

    def image_dir(self, split: str) -> Path:
        return self.root / self.splits[split]

    def label_dir(self, split: str) -> Path:
        parts = Path(self.splits[split]).parts
        return self.root / Path(*["labels" if p == "images" else p for p in parts])


def load_spec(data_yaml: str | Path, root: str | Path | None = None) -> DatasetSpec:
    data_yaml = Path(data_yaml)
    raw = yaml.safe_load(data_yaml.read_text())
    base = Path(root) if root else (data_yaml.parent / raw.get("path", ".")).resolve()
    names = raw["names"]
    if isinstance(names, list):
        names = dict(enumerate(names))
    splits = {k: raw[k] for k in ("train", "val", "test") if raw.get(k)}
    return DatasetSpec(root=base, splits=splits, names={int(k): str(v) for k, v in names.items()})


def parse_label_file(path: Path, num_classes: int) -> tuple[list[Label], list[LabelIssue]]:
    labels, issues = [], []
    seen = set()
    for i, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            issues.append(LabelIssue(path.name, i, f"expected 5 values, got {len(parts)}"))
            continue
        try:
            cls = int(parts[0])
            cx, cy, w, h = (float(p) for p in parts[1:])
        except ValueError:
            issues.append(LabelIssue(path.name, i, "non-numeric value"))
            continue
        if not 0 <= cls < num_classes:
            issues.append(LabelIssue(path.name, i, f"class {cls} out of range"))
            continue
        if w <= 0 or h <= 0:
            issues.append(LabelIssue(path.name, i, "zero or negative size"))
            continue
        if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)) or cx - w / 2 < -1e-3 or cx + w / 2 > 1 + 1e-3 \
                or cy - h / 2 < -1e-3 or cy + h / 2 > 1 + 1e-3:
            issues.append(LabelIssue(path.name, i, "box outside the image"))
            continue
        key = (cls, round(cx, 4), round(cy, 4), round(w, 4), round(h, 4))
        if key in seen:
            issues.append(LabelIssue(path.name, i, "duplicate box"))
            continue
        seen.add(key)
        labels.append(Label(cls, cx, cy, w, h))
    return labels, issues


def images(spec: DatasetSpec, split: str) -> list[Path]:
    d = spec.image_dir(split)
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS) if d.exists() else []


def label_path(spec: DatasetSpec, split: str, image: Path) -> Path:
    return spec.label_dir(split) / f"{image.stem}.txt"


@dataclass
class SplitStats:
    images: int = 0
    images_without_labels: int = 0
    boxes: int = 0
    class_counts: Counter = field(default_factory=Counter)
    boxes_per_image: list[int] = field(default_factory=list)
    rel_area: dict[int, list[float]] = field(default_factory=lambda: defaultdict(list))
    issues: list[LabelIssue] = field(default_factory=list)
    orphan_labels: list[str] = field(default_factory=list)


def scan(spec: DatasetSpec, split: str) -> SplitStats:
    st = SplitStats()
    imgs = images(spec, split)
    stems = {p.stem for p in imgs}
    for img in imgs:
        st.images += 1
        lp = label_path(spec, split, img)
        if not lp.exists():
            st.images_without_labels += 1
            st.boxes_per_image.append(0)
            continue
        labels, issues = parse_label_file(lp, len(spec.names))
        st.issues += issues
        st.boxes += len(labels)
        st.boxes_per_image.append(len(labels))
        for lb in labels:
            st.class_counts[lb.cls] += 1
            st.rel_area[lb.cls].append(lb.w * lb.h)
    ld = spec.label_dir(split)
    if ld.exists():
        st.orphan_labels = sorted(p.name for p in ld.glob("*.txt") if p.stem not in stems)
    return st


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dhash(path: Path, size: int = 8) -> int:
    """Difference hash: robust to resizing and recompression, used to find near-duplicates across splits."""
    from PIL import Image

    with Image.open(path) as im:
        im.draft("L", (size * 8, size * 8))          # fast JPEG decode at reduced resolution
        small = np.asarray(im.convert("L").resize((size + 1, size), Image.Resampling.BOX), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def leakage(spec: DatasetSpec, a: str = "train", b: str = "val", max_hamming: int = 4) -> dict:
    """Images that appear in both splits: identical bytes, or a near-identical picture (dHash distance)."""
    ha = {file_hash(p): p.name for p in images(spec, a)}
    hb = {file_hash(p): p.name for p in images(spec, b)}
    exact = sorted((ha[h], hb[h]) for h in set(ha) & set(hb))
    da = {p.name: dhash(p) for p in images(spec, a)}
    db = {p.name: dhash(p) for p in images(spec, b)}
    near = []
    va = np.array(list(da.values()), dtype=np.uint64)
    names_a = list(da)
    for nb, hv in db.items():
        x = va ^ np.uint64(hv)
        dist = np.array([bin(int(v)).count("1") for v in x])
        for i in np.where(dist <= max_hamming)[0]:
            pair = (names_a[i], nb)
            if pair not in exact:
                near.append((*pair, int(dist[i])))
    return {"exact": exact, "near": sorted(near)}
