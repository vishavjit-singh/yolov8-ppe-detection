"""Detection metrics (AP per class, mAP@0.5 and mAP@0.5:0.95) for checking an exported model.

Used in CI to evaluate the ONNX model through this repo's own inference path, as a cross-check of the
numbers Ultralytics reports for the PyTorch model. AP uses 101-point interpolation like COCO.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .inference import Detection, iou_matrix

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """COCO-style AP: mean over 101 recall levels of the best precision reached at that recall or beyond."""
    if len(recall) == 0:
        return 0.0
    envelope = np.flip(np.maximum.accumulate(np.flip(precision)))      # max precision at recall >= r
    levels = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, levels, side="left")
    values = np.where(idx < len(recall), envelope[np.minimum(idx, len(recall) - 1)], 0.0)
    return float(values.mean())


def evaluate(predictions: list[list[Detection]], ground_truth: list[list[Detection]], num_classes: int) -> dict:
    """``predictions[i]`` and ``ground_truth[i]`` belong to image i. Returns per-class and mean AP."""
    if len(predictions) != len(ground_truth):
        raise ValueError("predictions and ground truth must cover the same images")
    # per class: list of (score, image, box) and the GT boxes per image
    preds = defaultdict(list)
    gts: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    n_gt = np.zeros(num_classes, dtype=int)
    for img, (p_list, g_list) in enumerate(zip(predictions, ground_truth)):
        for d in p_list:
            preds[d.cls].append((d.score, img, d.box))
        by_cls = defaultdict(list)
        for g in g_list:
            by_cls[g.cls].append(g.box)
        for c, boxes in by_cls.items():
            gts[c][img] = np.array(boxes, dtype=float)
            n_gt[c] += len(boxes)
    ap = np.full((num_classes, len(IOU_THRESHOLDS)), np.nan)
    for c in range(num_classes):
        if n_gt[c] == 0:
            continue
        items = sorted(preds[c], key=lambda t: -t[0])
        if not items:
            ap[c] = 0.0
            continue
        for k, thr in enumerate(IOU_THRESHOLDS):
            used = {img: np.zeros(len(b), dtype=bool) for img, b in gts[c].items()}
            tp = np.zeros(len(items))
            for j, (_, img, box) in enumerate(items):
                g = gts[c].get(img)
                if g is None:
                    continue
                ious = iou_matrix(np.array([box]), g)[0]
                ious[used[img]] = -1
                m = int(np.argmax(ious))
                if ious[m] >= thr:
                    tp[j] = 1
                    used[img][m] = True
            ctp = np.cumsum(tp)
            recall = ctp / n_gt[c]
            precision = ctp / np.arange(1, len(items) + 1)
            ap[c, k] = average_precision(recall, precision)
    valid = ~np.isnan(ap[:, 0])
    return {
        "per_class": {c: {"ap50": float(ap[c, 0]), "ap50_95": float(np.mean(ap[c])), "instances": int(n_gt[c])}
                      for c in range(num_classes) if valid[c]},
        "map50": float(np.mean(ap[valid, 0])) if valid.any() else 0.0,
        "map50_95": float(np.mean(ap[valid])) if valid.any() else 0.0,
    }
