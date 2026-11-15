from pathlib import Path

import pytest

from ppe.compliance import load_policy
from ppe.dataset import load_spec
from ppe.inference import Detection
from ppe.synthetic import NAMES, make

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def policy():
    return load_policy(ROOT / "config" / "site_policy.yaml")


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory):
    return load_spec(make(tmp_path_factory.mktemp("syn") / "ppe"))


@pytest.fixture
def det():
    ids = {v: k for k, v in NAMES.items()}

    def make_det(name, box, score=0.9):
        return Detection(ids[name], name, score, tuple(float(v) for v in box))
    return make_det
