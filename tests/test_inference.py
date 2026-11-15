import numpy as np
import pytest

from ppe.inference import Letterbox, OnnxDetector, decode, iou_matrix, letterbox, nms, preprocess

NAMES = {0: "helmet", 1: "vest", 2: "Person"}


def test_letterbox_keeps_aspect_and_centres():
    img = np.zeros((300, 600, 3), np.uint8)
    out, lb = letterbox(img, 640)
    assert out.shape == (640, 640, 3)
    assert lb.scale == pytest.approx(640 / 600)
    assert lb.pad_x == 0 and lb.pad_y == 160
    assert (out[:160] == 114).all() and (out[160:480] == 0).all()


def test_preprocess_layout():
    img = np.zeros((10, 20, 3), np.uint8)
    img[..., 0] = 255                                  # blue in BGR
    x, _ = preprocess(img, 32)
    assert x.shape == (1, 3, 32, 32) and x.dtype == np.float32
    assert x[0, 2, 16, 16] == pytest.approx(1.0) and x[0, 0, 16, 16] == 0.0   # became the last (B) channel


def test_iou_and_nms():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]], dtype=float)
    assert iou_matrix(boxes[:1], boxes[1:2])[0, 0] == pytest.approx(81 / 119)
    assert list(nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5)) == [0, 2]
    assert list(nms(boxes, np.array([0.9, 0.8, 0.7]), 0.9)) == [0, 1, 2]


def raw_output(rows, nc=3, n=8):
    """Build a (1, 4+nc, n) YOLOv8 head output from (cx, cy, w, h, cls, score) rows."""
    out = np.zeros((1, 4 + nc, n), np.float32)
    for i, (cx, cy, w, h, c, s) in enumerate(rows):
        out[0, :4, i] = [cx, cy, w, h]
        out[0, 4 + c, i] = s
    return out


def test_decode_maps_back_to_the_original_image():
    lb = Letterbox(scale=0.5, pad_x=0, pad_y=80)       # e.g. a 1280×960 image into 640×640
    out = raw_output([(100, 180, 40, 40, 0, 0.9)])
    (d,) = decode(out, lb, (960, 1280), NAMES)
    assert d.name == "helmet" and d.score == pytest.approx(0.9)
    assert d.box == pytest.approx((160.0, 160.0, 240.0, 240.0))


def test_decode_threshold_classwise_nms_and_clipping():
    lb = Letterbox(1.0, 0, 0)
    out = raw_output([
        (50, 50, 20, 20, 0, 0.9),
        (51, 51, 20, 20, 0, 0.6),      # same class, overlapping → suppressed
        (51, 51, 20, 20, 1, 0.6),      # other class, same place → kept
        (5, 5, 30, 30, 2, 0.5),        # partly outside → clipped to 0
        (200, 200, 10, 10, 2, 0.1),    # below conf
    ])
    dets = decode(out, lb, (100, 100), NAMES, conf=0.25, iou=0.45)
    assert [(d.name, round(d.score, 2)) for d in dets] == [("helmet", 0.9), ("vest", 0.6), ("Person", 0.5)]
    assert dets[2].box[:2] == (0.0, 0.0)
    assert decode(raw_output([]), lb, (100, 100), NAMES) == []


class FakeInput:
    name, shape = "images", [1, 3, 64, 64]


class FakeOutput:
    shape = [1, 7, 8]


class FakeSession:
    def __init__(self, out):
        self.out, self.seen = out, None

    def get_inputs(self):
        return [FakeInput()]

    def get_outputs(self):
        return [FakeOutput()]

    def run(self, names, feeds):
        self.seen = feeds["images"].shape
        return [self.out]


def test_detector_end_to_end_with_a_fake_session(monkeypatch):
    import onnxruntime

    session = FakeSession(raw_output([(32, 32, 16, 16, 2, 0.8)]))
    monkeypatch.setattr(onnxruntime, "InferenceSession", lambda *a, **k: session)
    detector = OnnxDetector("model.onnx", NAMES)
    dets = detector(np.zeros((128, 128, 3), np.uint8))   # scale 0.5, no padding
    assert session.seen == (1, 3, 64, 64)
    assert len(dets) == 1 and dets[0].box == pytest.approx((48, 48, 80, 80))
    with pytest.raises(ValueError, match="model has 3 classes, names has 2"):
        OnnxDetector("model.onnx", {0: "a", 1: "b"})
