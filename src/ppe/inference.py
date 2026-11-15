"""YOLOv8 inference with ONNX Runtime and NumPy only (no PyTorch or Ultralytics at serving time).

The exported YOLOv8 detection head outputs one tensor of shape ``(1, 4 + num_classes, N)``: for each of
the N anchor points, a box (centre x, centre y, width, height in input pixels) followed by one
sigmoid score per class. There is no objectness score and NMS is not baked in, so it is done here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Detection:
    cls: int
    name: str
    score: float
    box: tuple[float, float, float, float]          # x1, y1, x2, y2 in original image pixels

    def to_dict(self) -> dict:
        return {"class_id": self.cls, "class": self.name, "score": round(self.score, 4),
                "box": [round(v, 1) for v in self.box]}


@dataclass(frozen=True)
class Letterbox:
    scale: float
    pad_x: float
    pad_y: float


def letterbox(image: np.ndarray, size: int = 640, color: int = 114) -> tuple[np.ndarray, Letterbox]:
    """Resize keeping aspect ratio and pad to ``size × size`` (Ultralytics' default preprocessing)."""
    import cv2

    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nh, nw) != (h, w) else image
    pad_x, pad_y = (size - nw) / 2, (size - nh) / 2
    top, left = int(round(pad_y - 0.1)), int(round(pad_x - 0.1))
    out = np.full((size, size, 3), color, dtype=np.uint8)
    out[top:top + nh, left:left + nw] = resized
    return out, Letterbox(scale, left, top)


def preprocess(image_bgr: np.ndarray, size: int) -> tuple[np.ndarray, Letterbox]:
    padded, lb = letterbox(image_bgr, size)
    x = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0      # BGR→RGB, HWC→CHW, 0..1
    return np.ascontiguousarray(x[None]), lb


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between xyxy boxes ``a`` (N, 4) and ``b`` (M, 4)."""
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy NMS; returns kept indices sorted by descending score."""
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = iou_matrix(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_threshold]
    return np.array(keep, dtype=int)


def decode(output: np.ndarray, lb: Letterbox, image_shape: tuple[int, int], names: dict[int, str],
           conf: float = 0.25, iou: float = 0.45, max_det: int = 300) -> list[Detection]:
    """Turn a raw ``(1, 4 + nc, N)`` YOLOv8 output into detections in original image coordinates."""
    pred = output[0] if output.ndim == 3 else output
    pred = pred.T                                                   # (N, 4 + nc)
    boxes_xywh, cls_scores = pred[:, :4], pred[:, 4:]
    cls = cls_scores.argmax(axis=1)
    scores = cls_scores[np.arange(len(cls)), cls]
    mask = scores >= conf
    if not mask.any():
        return []
    b, s, c = boxes_xywh[mask], scores[mask], cls[mask]
    xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
    # class-aware NMS: offset boxes per class so different classes never suppress each other
    offset = c[:, None].astype(np.float32) * 4096.0
    keep = nms(xyxy + offset, s, iou)[:max_det]
    h, w = image_shape
    out = []
    for i in keep:
        x1, y1, x2, y2 = xyxy[i]
        x1, x2 = (np.clip([(x1 - lb.pad_x) / lb.scale, (x2 - lb.pad_x) / lb.scale], 0, w)).tolist()
        y1, y2 = (np.clip([(y1 - lb.pad_y) / lb.scale, (y2 - lb.pad_y) / lb.scale], 0, h)).tolist()
        out.append(Detection(int(c[i]), names.get(int(c[i]), str(int(c[i]))), float(s[i]), (x1, y1, x2, y2)))
    return out


class OnnxDetector:
    """A YOLOv8 ONNX model (``yolo export format=onnx``) served with ONNX Runtime on CPU."""

    def __init__(self, model_path: str | Path, names: dict[int, str], conf: float = 0.25, iou: float = 0.45):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.size = int(inp.shape[2]) if isinstance(inp.shape[2], int) else 640
        nc = self.session.get_outputs()[0].shape[1]
        if isinstance(nc, int) and nc - 4 != len(names):
            raise ValueError(f"model has {nc - 4} classes, names has {len(names)}")
        self.names, self.conf, self.iou = names, conf, iou

    def __call__(self, image_bgr: np.ndarray) -> list[Detection]:
        x, lb = preprocess(image_bgr, self.size)
        (out,) = self.session.run(None, {self.input_name: x})[:1]
        return decode(out, lb, image_bgr.shape[:2], self.names, self.conf, self.iou)
