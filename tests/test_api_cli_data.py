import json
import zipfile

import cv2
import numpy as np
import pytest
from click.testing import CliRunner
from starlette.testclient import TestClient

from ppe.api import create_app
from ppe.cli import cli
from ppe.data import download, safe_extract
from ppe.inference import Detection


def jpeg():
    ok, buf = cv2.imencode(".jpg", np.full((60, 80, 3), 200, np.uint8))
    return buf.tobytes()


def test_api_without_model_is_unavailable(monkeypatch):
    monkeypatch.delenv("PPE_MODEL", raising=False)
    c = TestClient(create_app())
    assert c.get("/health").json() == {"status": "ok", "model_loaded": False, "required_ppe": ["helmet", "vest"]}
    assert c.post("/v1/detect", files={"image": ("a.jpg", jpeg(), "image/jpeg")}).status_code == 503


def test_api_detect_with_a_stub_detector():
    def detector(img):
        return [Detection(5, "Person", 0.9, (10, 5, 40, 58)), Detection(6, "no_helmet", 0.8, (18, 5, 32, 15))]

    c = TestClient(create_app(detector=detector))
    r = c.post("/v1/detect", files={"image": ("a.jpg", jpeg(), "image/jpeg")})
    body = r.json()
    assert r.status_code == 200 and body["status"] == "violation" and body["width"] == 80
    assert body["persons"][0]["required"]["helmet"] == "violation"
    assert c.post("/v1/detect", files={"image": ("a.jpg", b"not an image", "image/jpeg")}).status_code == 422
    assert c.post("/v1/detect", data={"x": "1"}).status_code == 422


def test_safe_extract_rejects_zip_slip(tmp_path):
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("../outside.txt", "x")
    with pytest.raises(ValueError, match="unsafe path"):
        safe_extract(z, tmp_path / "out")


def test_download_verifies_checksum(tmp_path):
    z = tmp_path / "construction-ppe.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("construction-ppe/data.yaml", "names: {}\n")
    with pytest.raises(ValueError, match="checksum"):
        download(tmp_path / "d", zip_path=z)
    assert (download(tmp_path / "d", zip_path=z, expected_sha256=None) / "data.yaml").exists()


def test_cli_validate_and_demo_on_synthetic_data(synthetic, tmp_path):
    r = CliRunner()
    data = str(synthetic.root / "data.yaml")
    res = r.invoke(cli, ["validate", "--data", data])
    assert res.exit_code == 0, res.output
    assert "train: 12 images" in res.output and "0 identical files" in res.output
    res = r.invoke(cli, ["demo", "--data", data, "--out", str(tmp_path / "img"), "--summary", str(tmp_path / "s.json")])
    assert res.exit_code == 0, res.output
    s = json.loads((tmp_path / "s.json").read_text())
    assert s["images"] == {"train": 12, "val": 6} and s["label_issues"] == 0
    assert set(s["val_people_status"]) <= {"ok", "violation", "unverified"} and s["val_people"] > 0
    assert s["alert_replay"]["debounced_alerts"] >= 1
    assert len(list((tmp_path / "img").glob("*.png"))) == 5


def test_cli_validate_fails_on_bad_labels(synthetic, tmp_path):
    from ppe.synthetic import make

    data = make(tmp_path / "bad", train=2, val=1)
    lbl = next((tmp_path / "bad" / "labels" / "train").glob("*.txt"))
    lbl.write_text(lbl.read_text() + "99 0.5 0.5 0.1 0.1\n")
    res = CliRunner().invoke(cli, ["validate", "--data", str(data), "--no-leakage"])
    assert res.exit_code == 1 and "class 99 out of range" in res.output
