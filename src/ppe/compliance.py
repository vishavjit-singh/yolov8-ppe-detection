"""Site compliance: turn detections into per-person PPE verdicts.

A detector only says "there is a helmet here". A site needs "worker 3 has no helmet". This module:

1. associates every gear box (helmet, vest, …, and the explicit ``no_*`` classes) with a person, using
   how much of the gear box lies inside the person box (IoA) and whether its centre falls in the
   expected body region (helmet and goggles near the top, vest on the torso, boots at the bottom)
2. decides each required item per person: ``ok`` (gear seen), ``violation`` (the ``no_*`` class seen)
   or ``unverified`` (neither, for example when occluded). The dataset documentation is explicit that
   occlusion alone is not evidence of missing equipment, so ``unverified`` is not a violation unless
   the policy says so
3. rolls people up into a frame verdict that the alerting layer can debounce over time
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .inference import Detection

OK, VIOLATION, UNVERIFIED = "ok", "violation", "unverified"


@dataclass(frozen=True)
class ItemRule:
    name: str
    positive: str
    negative: str | None
    region: tuple[float, float]


@dataclass(frozen=True)
class Policy:
    person_class: str
    required: tuple[str, ...]
    optional: tuple[str, ...]
    items: dict[str, ItemRule]
    min_ioa: float
    region_margin: float
    missing_is_violation: bool
    raise_after: int
    window: int
    clear_after: int

    @property
    def gear_classes(self) -> dict[str, tuple[str, bool]]:
        """class name → (item, is_negative)"""
        out = {}
        for item in self.items.values():
            out[item.positive] = (item.name, False)
            if item.negative:
                out[item.negative] = (item.name, True)
        return out


def load_policy(path: str | Path) -> Policy:
    raw = yaml.safe_load(Path(path).read_text())
    items = {k: ItemRule(k, v["positive"], v.get("negative"), tuple(v["region"])) for k, v in raw["items"].items()}
    for name in list(raw["required"]) + list(raw.get("optional", [])):
        if name not in items:
            raise ValueError(f"policy item {name!r} has no rule")
    a, al = raw["association"], raw["alerts"]
    if al["raise_after"] > al["window"]:
        raise ValueError("alerts.raise_after cannot exceed alerts.window")
    return Policy(person_class=raw["person_class"], required=tuple(raw["required"]),
                  optional=tuple(raw.get("optional", [])), items=items, min_ioa=float(a["min_ioa"]),
                  region_margin=float(a["region_margin"]), missing_is_violation=bool(raw["missing_is_violation"]),
                  raise_after=int(al["raise_after"]), window=int(al["window"]), clear_after=int(al["clear_after"]))


def ioa(inner: tuple, outer: tuple) -> float:
    """Share of ``inner``'s area that lies inside ``outer``."""
    x1, y1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    x2, y2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = max((inner[2] - inner[0]) * (inner[3] - inner[1]), 1e-9)
    return inter / area


def in_region(gear: tuple, person: tuple, region: tuple[float, float], margin: float) -> bool:
    cx = (gear[0] + gear[2]) / 2
    cy = (gear[1] + gear[3]) / 2
    h = person[3] - person[1]
    top = person[1] + (region[0] - margin) * h
    bottom = person[1] + (region[1] + margin) * h
    return person[0] <= cx <= person[2] and top <= cy <= bottom


@dataclass
class PersonVerdict:
    index: int
    box: tuple[float, float, float, float]
    score: float
    items: dict[str, str]                         # item → ok / violation / unverified
    gear: list[Detection] = field(default_factory=list)

    @property
    def status(self) -> str:
        return VIOLATION if VIOLATION in self.items_required.values() else (
            UNVERIFIED if UNVERIFIED in self.items_required.values() else OK)

    items_required: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"person": self.index, "box": [round(v, 1) for v in self.box], "status": self.status,
                "required": self.items_required, "items": self.items}


@dataclass
class FrameVerdict:
    people: list[PersonVerdict]
    unassigned: list[Detection]

    @property
    def violations(self) -> int:
        return sum(1 for p in self.people if p.status == VIOLATION)

    @property
    def status(self) -> str:
        if self.violations:
            return VIOLATION
        return UNVERIFIED if any(p.status == UNVERIFIED for p in self.people) else OK

    def to_dict(self) -> dict:
        return {"status": self.status, "people": len(self.people), "violations": self.violations,
                "unassigned_gear": len(self.unassigned), "persons": [p.to_dict() for p in self.people]}


def evaluate(detections: list[Detection], policy: Policy) -> FrameVerdict:
    persons = [d for d in detections if d.name == policy.person_class]
    persons.sort(key=lambda d: (d.box[0], d.box[1]))
    gear_map = policy.gear_classes
    gear = [d for d in detections if d.name in gear_map]
    assigned: dict[int, list[Detection]] = {i: [] for i in range(len(persons))}
    unassigned = []
    for g in gear:
        item, _ = gear_map[g.name]
        rule = policy.items[item]
        best, best_score = None, 0.0
        for i, p in enumerate(persons):
            share = ioa(g.box, p.box)
            if share >= policy.min_ioa and in_region(g.box, p.box, rule.region, policy.region_margin):
                # prefer the smallest enclosing person: in crowds the nearest worker, not the one behind
                area = (p.box[2] - p.box[0]) * (p.box[3] - p.box[1])
                score = share / max(area, 1.0) ** 0.25
                if score > best_score:
                    best, best_score = i, score
        if best is None:
            unassigned.append(g)
        else:
            assigned[best].append(g)
    people = []
    for i, p in enumerate(persons):
        items = {}
        for name in policy.required + policy.optional:
            rule = policy.items[name]
            pos = [g for g in assigned[i] if g.name == rule.positive]
            neg = [g for g in assigned[i] if rule.negative and g.name == rule.negative]
            if neg and (not pos or max(n.score for n in neg) > max(q.score for q in pos)):
                items[name] = VIOLATION
            elif pos:
                items[name] = OK
            else:
                items[name] = VIOLATION if policy.missing_is_violation else UNVERIFIED
        v = PersonVerdict(index=i, box=p.box, score=p.score, items=items, gear=assigned[i])
        v.items_required = {k: items[k] for k in policy.required}
        people.append(v)
    return FrameVerdict(people=people, unassigned=unassigned)
