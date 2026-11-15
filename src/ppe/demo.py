"""`make demo`: dataset analysis + the compliance engine on ground-truth labels + an alert-debouncing replay.

No model is involved here (training runs in CI). Everything is computed from the annotated dataset, so the
numbers describe the data and the compliance logic, not detector accuracy.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402

from .alerts import AlertDebouncer  # noqa: E402
from .compliance import OK, UNVERIFIED, VIOLATION, Policy, evaluate  # noqa: E402
from .dataset import DatasetSpec, images, leakage, scan  # noqa: E402
from .draw import annotate  # noqa: E402
from .gt import gt_detections  # noqa: E402
from .simulate import stream  # noqa: E402

INK, MUTED, GRID = "#0e1a2b", "#5b6676", "#e3e7ee"
ACCENT, SIGNAL, BAD, GOOD, PURPLE = "#2a78d6", "#eda100", "#e34948", "#1baf7a", "#6250d6"
STATUS_COLORS = {OK: GOOD, VIOLATION: BAD, UNVERIFIED: SIGNAL}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.titleweight": "bold", "axes.titlesize": 13, "axes.titlecolor": INK,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})


def _save(fig, out: Path, name: str) -> Path:
    fig.tight_layout()
    p = out / name
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    return p


# ------------------------------------------------------------------------------------------------ analysis
def analyse(spec: DatasetSpec, policy: Policy) -> dict:
    stats = {s: scan(spec, s) for s in spec.splits}
    leak = leakage(spec, "train", "val") if {"train", "val"} <= set(spec.splits) else {"exact": [], "near": []}
    val_imgs = images(spec, "val")
    frames = []
    for img in val_imgs:
        dets = gt_detections(spec, "val", img)
        frames.append((img, dets, evaluate(dets, policy)))
    people = [p for _, _, v in frames for p in v.people]
    item_counts = {item: Counter(p.items[item] for p in people) for item in policy.required + policy.optional}
    return {"stats": stats, "leakage": leak, "frames": frames, "people": people, "item_counts": item_counts}


def summary(spec: DatasetSpec, policy: Policy, a: dict) -> dict:
    st = a["stats"]
    people = a["people"]
    return {
        "dataset": "Ultralytics Construction-PPE (CC BY 4.0)",
        "images": {s: v.images for s, v in st.items()},
        "boxes": {s: v.boxes for s, v in st.items()},
        "classes": len(spec.names),
        "label_issues": sum(len(v.issues) for v in st.values()),
        "images_without_labels": sum(v.images_without_labels for v in st.values()),
        "orphan_label_files": sum(len(v.orphan_labels) for v in st.values()),
        "leakage_exact": len(a["leakage"]["exact"]),
        "leakage_near_duplicates": len(a["leakage"]["near"]),
        "val_people": len(people),
        "val_people_status": dict(Counter(p.status for p in people)),
        "val_required_items": {k: dict(v) for k, v in a["item_counts"].items() if k in policy.required},
        "val_unassigned_gear": sum(len(v.unassigned) for _, _, v in a["frames"]),
        "val_frames_with_violation": sum(1 for _, _, v in a["frames"] if v.status == VIOLATION),
    }


# -------------------------------------------------------------------------------------------------- charts
def architecture(out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.0)
    ax.axis("off")

    def box(x, y, w, h, title, sub, color=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc="white",
                                    ec=color, lw=2))
        ax.text(x + w / 2, y + h * 0.7, title, ha="center", va="center", fontsize=11, weight="bold", color=color)
        ax.text(x + w / 2, y + h * 0.32, sub, ha="center", va="center", fontsize=8.2, color=MUTED, linespacing=1.4)

    def arrow(x1, y1, x2, y2, color=MUTED):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, lw=1.6, color=color))

    ax.text(0.15, 5.7, "yolov8-ppe-detection: train in CI, serve with ONNX Runtime, decide per worker",
            fontsize=13.5, weight="bold", color=INK)
    ax.text(0.15, 5.3, "TRAINING (GitHub Actions, CPU)", fontsize=8.5, color=MUTED, weight="bold")
    box(0.2, 3.6, 2.3, 1.45, "Construction-PPE", "1,000 images · 12 classes\nCC BY 4.0 · sha256 pinned", ACCENT)
    box(2.9, 3.6, 2.3, 1.45, "Dataset checks", "label validation · stats\ntrain/val leakage (dHash)", INK)
    box(5.6, 3.6, 2.4, 1.45, "Ultralytics YOLOv8n", "yolo train (pinned)\nval → mAP per class\nexport → ONNX", PURPLE)
    box(8.4, 3.6, 2.2, 1.45, "Artifacts", "best.onnx · metrics\nprediction images", SIGNAL)
    ax.text(0.15, 3.05, "SERVING (no PyTorch, no Ultralytics)", fontsize=8.5, color=MUTED, weight="bold")
    box(0.2, 0.9, 2.3, 1.6, "Camera / image", "frames or uploads", INK)
    box(2.9, 0.9, 2.6, 1.6, "OnnxDetector", "letterbox → ONNX Runtime\ndecode (4+nc)×N\nclass-aware NMS",
        PURPLE)
    box(5.9, 0.9, 2.6, 1.6, "Compliance engine", "gear → person by IoA\n+ body region\nok / violation / unverified",
        GOOD)
    box(8.9, 0.9, 2.0, 1.6, "Alerts", "N-of-M debounce\nevent log", BAD)
    box(11.1, 0.9, 1.75, 1.6, "API · CLI", "Starlette\n/v1/detect", INK)
    arrow(2.5, 4.32, 2.9, 4.32)
    arrow(5.2, 4.32, 5.6, 4.32)
    arrow(8.0, 4.32, 8.4, 4.32)
    arrow(9.5, 3.6, 4.2, 2.5, SIGNAL)
    arrow(2.5, 1.7, 2.9, 1.7)
    arrow(5.5, 1.7, 5.9, 1.7)
    arrow(8.5, 1.7, 8.9, 1.7)
    arrow(10.9, 1.7, 11.1, 1.7)
    ax.text(7.3, 3.2, "best.onnx", fontsize=8.5, color="#9a6a00")
    ax.text(0.2, 0.35, "Site policy (config/site_policy.yaml): helmet + vest required; gloves, boots, goggles "
            "reported. No gear and no no_* box → 'unverified', because occlusion is not evidence of a violation.",
            fontsize=8.3, color=MUTED)
    return _save(fig, out, "architecture.png")


def dataset_stats(spec: DatasetSpec, a: dict, out: Path) -> Path:
    st = a["stats"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    ids = sorted(spec.names, key=lambda c: -sum(s.class_counts[c] for s in st.values()))
    y = np.arange(len(ids))
    left = np.zeros(len(ids))
    for split, color in (("train", ACCENT), ("val", PURPLE)):
        if split not in st:
            continue
        vals = np.array([st[split].class_counts[c] for c in ids])
        a1.barh(y, vals, left=left, color=color, height=0.65, label=f"{split} ({st[split].images} images)")
        left += vals
    for i, v in enumerate(left):
        a1.text(v, i, f" {int(v):,}", va="center", fontsize=8.5, color=INK)
    a1.set_yticks(y, [spec.names[c] for c in ids])
    a1.invert_yaxis()
    a1.grid(axis="x", color=GRID)
    a1.set_xlim(0, left.max() * 1.15)
    a1.legend(frameon=False, fontsize=8.5, loc="lower right")
    total = sum(s.boxes for s in st.values())
    a1.set_title(f"Boxes per class ({total:,} boxes, 12 classes)")
    data = [np.sqrt(np.array(sum((s.rel_area[c] for s in st.values()), []))) * 100 for c in ids]
    a2.boxplot(data, orientation="horizontal", widths=0.6, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="#dbe8f8", edgecolor=ACCENT), medianprops=dict(color=INK))
    a2.set_yticks(range(1, len(ids) + 1), [spec.names[c] for c in ids])
    a2.invert_yaxis()
    a2.set_xscale("log")
    a2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}%"))
    a2.grid(axis="x", color=GRID)
    a2.set_xlabel("box size: √(w·h) as % of the image side (log)")
    a2.set_title("Object size per class")
    return _save(fig, out, "dataset_stats.png")


def compliance_summary(policy: Policy, a: dict, out: Path) -> Path:
    items = list(policy.required + policy.optional)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 3.9), gridspec_kw={"width_ratios": [1.6, 1]})
    y = np.arange(len(items))
    left = np.zeros(len(items))
    for status in (OK, VIOLATION, UNVERIFIED):
        vals = np.array([a["item_counts"][i][status] for i in items])
        a1.barh(y, vals, left=left, color=STATUS_COLORS[status], height=0.62, label=status)
        for k, v in enumerate(vals):
            if v >= 12:
                a1.text(left[k] + v / 2, k, str(v), ha="center", va="center", fontsize=8.5,
                        color="white" if status != UNVERIFIED else INK)
        left += vals
    a1.set_yticks(y, [f"{i} {'(required)' if i in policy.required else ''}" for i in items])
    a1.invert_yaxis()
    a1.grid(axis="x", color=GRID)
    a1.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    a1.set_title(f"Per-item verdicts · {len(a['people'])} workers (val, ground truth)")
    st = Counter(p.status for p in a["people"])
    labels = [OK, VIOLATION, UNVERIFIED]
    a2.bar(labels, [st[s] for s in labels], color=[STATUS_COLORS[s] for s in labels], width=0.6)
    for i, s in enumerate(labels):
        a2.text(i, st[s], f"{st[s]}", ha="center", va="bottom", fontsize=9)
    a2.grid(axis="y", color=GRID)
    a2.set_title("Worker verdict (helmet + vest policy)")
    return _save(fig, out, "compliance_summary.png")


def pick_examples(a: dict, k: int = 3) -> list:
    """Frames with 2–4 workers that show at least one violation and one compliant worker."""
    def person_height(f) -> float:
        from .gt import image_size

        h = image_size(f[0])[1]
        return float(np.median([(p.box[3] - p.box[1]) / h for p in f[2].people]))

    good = [f for f in a["frames"] if 2 <= len(f[2].people) <= 3 and f[2].violations
            and any(p.status == OK for p in f[2].people)]
    good.sort(key=lambda f: (-round(person_height(f), 1), f[0].name))
    if not good:                                    # tiny or unusual datasets: any frame with workers
        good = [f for f in a["frames"] if f[2].people]
    return good[:k]


def compliance_examples(a: dict, out: Path) -> Path:
    ex = pick_examples(a)
    fig, axes = plt.subplots(1, len(ex), figsize=(13, 4.1))
    for ax, (img, dets, verdict) in zip(np.atleast_1d(axes), ex):
        im = cv2.imread(str(img))
        scale = 900 / max(im.shape[:2])
        im = annotate(im, verdict)
        im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ax.imshow(im[:, :, ::-1])
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{len(verdict.people)} workers · {verdict.violations} in violation", fontsize=10.5)
    fig.legend(handles=[Patch(color=GOOD, label="compliant"), Patch(color=BAD, label="violation (no_helmet / no_vest)"),
                        Patch(color=SIGNAL, label="unverified (gear not visible)"),
                        Patch(color=ACCENT, label="gear box assigned to the worker")],
               loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Compliance engine on ground-truth labels (Construction-PPE val split, CC BY 4.0)", x=0.01,
                 ha="left", fontsize=13, weight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    p = out / "compliance_examples.png"
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    return p


def alert_replay(a: dict, policy: Policy, out: Path, fps: float = 15.0) -> tuple[Path, dict]:
    # a frame where every worker wears a helmet and a vest, so the incident is the only real violation
    cands = [f for f in a["frames"] if 2 <= len(f[2].people) <= 5 and all(
        p.items_required.get("helmet") == OK and p.items_required.get("vest") == OK for p in f[2].people)]
    cands.sort(key=lambda f: f[0].name)
    if not cands:
        cands = [f for f in a["frames"] if any(p.status == OK for p in f[2].people)]
    img, dets, verdict = cands[0]
    target = next(p.box for p in verdict.people if p.status == OK)
    deb = AlertDebouncer(policy.raise_after, policy.window, policy.clear_after)
    rows, raw_transitions, prev = [], 0, False
    for f, frame_dets in stream(dets, incident_person_box=target):
        v = evaluate(frame_dets, policy)
        bad = v.violations > 0
        raw_transitions += int(bad and not prev)
        prev = bad
        reasons = Counter(k for p in v.people for k, s in p.items_required.items() if s == VIOLATION)
        deb.update(f, v.violations, dict(reasons))
        rows.append((f, v.violations, deb.active))
    deb.close(rows[-1][0])
    fig, ax = plt.subplots(figsize=(12, 3.4))
    f = np.array([r[0] for r in rows]) / fps
    ax.bar(f, [r[1] for r in rows], width=1 / fps, color=BAD, alpha=0.75, label="workers in violation (raw, per frame)")
    for e in deb.events:
        ax.axvspan(e.start_frame / fps, (e.end_frame + 1) / fps, color=SIGNAL, alpha=0.22, lw=0)
        ax.text(e.start_frame / fps + 0.05, 2.15, f"ALERT {e.start_frame / fps:.1f}–{(e.end_frame + 1) / fps:.1f} s",
                fontsize=8.5, color="#9a6a00", weight="bold")
    ax.axvspan(55 / fps, 95 / fps, ymin=0, ymax=0.04, color=INK)
    ax.text(55 / fps, -0.32, "real incident: helmet off (frames 55–95)", fontsize=8, color=INK)
    ax.set_ylim(-0.5, 2.5)
    ax.set_yticks([0, 1, 2])
    ax.set_ylabel("workers")
    ax.set_xlabel("seconds (15 fps)")
    ax.grid(axis="x", color=GRID)
    alert_label = (f"debounced alert ({policy.raise_after} of {policy.window} frames, "
                   f"clear after {policy.clear_after})")
    ax.legend(handles=[Patch(color=BAD, alpha=0.75, label="raw per-frame violation"),
                       Patch(color=SIGNAL, alpha=0.3, label=alert_label)],
              frameon=False, fontsize=8.5, loc="upper right")
    ax.set_title("Alert debouncing on a simulated stream (real labels + injected misses and false positives)")
    p = _save(fig, out, "alert_timeline.png")
    return p, {"frames": len(rows), "raw_alerts_without_debounce": raw_transitions,
               "debounced_alerts": len(deb.events), "events": [e.to_dict(fps) for e in deb.events],
               "source_image": img.name}


def run(spec: DatasetSpec, policy: Policy, out: Path, summary_path: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    a = analyse(spec, policy)
    s = summary(spec, policy, a)
    architecture(out)
    dataset_stats(spec, a, out)
    compliance_summary(policy, a, out)
    compliance_examples(a, out)
    _, replay = alert_replay(a, policy, out)
    s["alert_replay"] = replay
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(s, indent=2))
    return s
