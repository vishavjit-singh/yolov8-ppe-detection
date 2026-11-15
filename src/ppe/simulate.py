"""A simulated detection stream for exercising alert debouncing.

Real footage is not part of the dataset (it has still images), so the demo takes the ground-truth boxes
of one annotated image and replays them as a 15 fps "video" with seeded detector noise:

* each gear box is missed with probability ``miss_rate`` in every frame (flicker)
* a compliant worker occasionally shows a spurious ``no_helmet`` (false positive)
* during ``incident`` frames one worker really takes the helmet off (helmet → no_helmet)

Everything here is synthetic noise on top of real annotations, and labelled as such in the outputs.
"""
from __future__ import annotations

import random
from dataclasses import replace

from .inference import Detection


def stream(base: list[Detection], frames: int = 150, miss_rate: float = 0.12, false_positive_rate: float = 0.04,
           incident: tuple[int, int] = (55, 95), incident_person_box: tuple | None = None, seed: int = 3,
           names: dict[str, int] | None = None):
    rng = random.Random(seed)
    no_helmet_id = (names or {}).get("no_helmet", 6)
    for f in range(frames):
        dets = []
        for d in base:
            if d.name == "helmet" and incident_person_box is not None and incident[0] <= f < incident[1] \
                    and _inside(d.box, incident_person_box):
                dets.append(replace(d, cls=no_helmet_id, name="no_helmet", score=0.8))
                continue
            if d.name != "Person" and rng.random() < miss_rate:
                continue
            jitter = [rng.gauss(0, 1.5) for _ in range(4)]
            box = tuple(v + j for v, j in zip(d.box, jitter))
            dets.append(replace(d, box=box, score=round(rng.uniform(0.55, 0.95), 3)))
        if rng.random() < false_positive_rate:
            helmets = [d for d in dets if d.name == "helmet"]
            if helmets:
                h = rng.choice(helmets)
                dets.append(replace(h, cls=no_helmet_id, name="no_helmet", score=0.9))
                dets.remove(h)
        yield f, dets


def _inside(box: tuple, outer: tuple) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]
