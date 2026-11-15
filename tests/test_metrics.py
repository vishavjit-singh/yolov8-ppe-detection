import pytest

from ppe.inference import Detection
from ppe.metrics import evaluate


def d(cls, box, score=1.0):
    return Detection(cls, str(cls), score, box)


GT = [[d(0, (0, 0, 10, 10)), d(1, (20, 20, 40, 40))], [d(0, (50, 50, 70, 70))]]


def test_perfect_predictions():
    m = evaluate(GT, GT, num_classes=2)
    assert m["map50"] == pytest.approx(1.0) and m["map50_95"] == pytest.approx(1.0)
    assert m["per_class"][0]["instances"] == 2


def test_missed_and_false_positives_lower_ap():
    preds = [[d(0, (0, 0, 10, 10), 0.9), d(1, (100, 100, 120, 120), 0.95)], []]
    m = evaluate(preds, GT, num_classes=2)
    assert m["per_class"][0]["ap50"] == pytest.approx(0.5, abs=0.01)   # one of two helmets found
    assert m["per_class"][1]["ap50"] == 0.0                              # wrong place


def test_loose_boxes_count_at_iou_50_but_not_at_95():
    preds = [[d(0, (0, 0, 12, 12)), d(1, (20, 20, 40, 40))], [d(0, (50, 50, 70, 70))]]
    m = evaluate(preds, GT, num_classes=2)
    assert m["per_class"][0]["ap50"] == pytest.approx(1.0)
    assert m["per_class"][0]["ap50_95"] < 1.0


def test_duplicate_detections_are_false_positives():
    preds = [[d(0, (0, 0, 10, 10), 0.9), d(0, (0, 0, 10, 10), 0.8), d(1, (20, 20, 40, 40))], [d(0, (50, 50, 70, 70))]]
    assert evaluate(preds, GT, 2)["per_class"][0]["ap50"] == pytest.approx(1.0)  # recall reached before the dup


def test_classes_without_ground_truth_are_ignored():
    m = evaluate([[]], [[d(1, (0, 0, 5, 5))]], num_classes=3)
    assert set(m["per_class"]) == {1} and m["map50"] == 0.0
