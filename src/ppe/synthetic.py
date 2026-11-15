"""A tiny SYNTHETIC YOLO dataset (drawn shapes, not photos) for tests and offline smoke runs.

Each image has 1–3 "workers": a person box with a head, torso and feet region, and gear boxes placed
on those regions. Some workers get ``no_helmet`` / ``no_vest`` instead, some get nothing (occluded).
"""
from __future__ import annotations

import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

NAMES = {0: "helmet", 1: "gloves", 2: "vest", 3: "boots", 4: "goggles", 5: "Person", 6: "no_helmet",
         7: "no_goggle", 8: "no_gloves", 9: "no_boots", 10: "forklift", 11: "no_vest"}


def _yolo(cls, x1, y1, x2, y2, w, h) -> str:
    return f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


def make(root: Path, train: int = 12, val: int = 6, seed: int = 0, size: tuple[int, int] = (320, 240)) -> Path:
    rng = random.Random(seed)
    if root.exists():
        shutil.rmtree(root)
    w, h = size
    for split, n in (("train", train), ("val", val)):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for i in range(n):
            img = np.full((h, w, 3), (rng.randrange(150, 230),) * 3, np.uint8)
            cv2.circle(img, (rng.randrange(w), rng.randrange(h)), rng.randrange(10, 60), (rng.randrange(255),) * 3, -1)
            lines = []
            workers = rng.randint(1, 3)
            for k in range(workers):
                pw, ph = 50, 150
                x1 = 10 + k * 100 + rng.randrange(0, 20)
                y1 = rng.randrange(20, h - ph - 5)
                x2, y2 = x1 + pw, y1 + ph
                cv2.rectangle(img, (x1, y1), (x2, y2), (60, 60, 60), 2)
                lines.append(_yolo(5, x1, y1, x2, y2, w, h))
                kind = (i + k) % 4            # 0 compliant, 1 no helmet, 2 no vest, 3 occluded (no gear boxes)
                if kind != 3:
                    hc = 6 if kind == 1 else 0
                    cv2.circle(img, ((x1 + x2) // 2, y1 + 15), 12, (0, 200, 255) if hc == 0 else (90, 120, 160), -1)
                    lines.append(_yolo(hc, x1 + 12, y1 + 2, x2 - 12, y1 + 30, w, h))
                    vc = 11 if kind == 2 else 2
                    vest_color = (0, 140, 255) if vc == 2 else (120, 90, 60)
                    cv2.rectangle(img, (x1 + 6, y1 + 40), (x2 - 6, y1 + 95), vest_color, -1)
                    lines.append(_yolo(vc, x1 + 6, y1 + 40, x2 - 6, y1 + 95, w, h))
                    lines.append(_yolo(3, x1 + 4, y2 - 18, x2 - 4, y2 - 1, w, h))
            name = f"syn_{split}_{i:03d}"
            cv2.imwrite(str(root / "images" / split / f"{name}.jpg"), img)
            (root / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + "\n")
    data = {"path": str(root), "train": "images/train", "val": "images/val", "names": NAMES}
    (root / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    return root / "data.yaml"
