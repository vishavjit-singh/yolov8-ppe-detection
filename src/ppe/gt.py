"""Ground-truth labels as detections (score 1.0), so the compliance engine can run on annotated data."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from .dataset import DatasetSpec, label_path, parse_label_file
from .inference import Detection


def image_size(path: Path) -> tuple[int, int]:
    """(width, height) from the file header, without decoding the pixels."""
    with Image.open(path) as im:
        return im.size


def gt_detections(spec: DatasetSpec, split: str, image: Path) -> list[Detection]:
    w, h = image_size(image)
    lp = label_path(spec, split, image)
    if not lp.exists():
        return []
    labels, _ = parse_label_file(lp, len(spec.names))
    return [Detection(lb.cls, spec.names[lb.cls], 1.0, lb.xyxy(w, h)) for lb in labels]
