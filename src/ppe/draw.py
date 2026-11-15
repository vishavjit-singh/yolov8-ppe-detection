"""Annotated frames: person boxes coloured by verdict, gear boxes labelled."""
from __future__ import annotations

import cv2
import numpy as np

from .compliance import OK, UNVERIFIED, VIOLATION, FrameVerdict
from .inference import Detection

# BGR
STATUS_COLOR = {OK: (122, 175, 27), VIOLATION: (72, 73, 227), UNVERIFIED: (0, 161, 237)}
GEAR_COLOR = (214, 120, 42)
NEG_COLOR = (72, 73, 227)


def annotate(image: np.ndarray, verdict: FrameVerdict, detections: list[Detection] | None = None) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    t = max(2, int(round(min(h, w) / 300)))
    fs = max(0.6, min(h, w) / 700)
    for p in verdict.people:
        for g in p.gear:
            x1, y1, x2, y2 = (int(v) for v in g.box)
            color = NEG_COLOR if g.name.startswith("no_") else GEAR_COLOR
            cv2.rectangle(out, (x1, y1), (x2, y2), color, max(1, t - 1))
    for p in verdict.people:
        x1, y1, x2, y2 = (int(v) for v in p.box)
        color = STATUS_COLOR[p.status]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, t + 1)
        bad = [k for k, v in p.items_required.items() if v == VIOLATION]
        unk = [k for k, v in p.items_required.items() if v == UNVERIFIED]
        text = f"#{p.index} " + ("OK" if p.status == OK else (
            "NO " + "+".join(bad) if bad else "? " + "+".join(unk)))
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, t)
        y = max(th + 6, y1)
        cv2.rectangle(out, (x1, y - th - 8), (x1 + tw + 8, y), color, -1)
        cv2.putText(out, text, (x1 + 4, y - 5), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), t, cv2.LINE_AA)
    return out
