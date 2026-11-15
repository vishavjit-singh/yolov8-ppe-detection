import pytest
import yaml

from ppe.compliance import OK, UNVERIFIED, VIOLATION, evaluate, ioa, load_policy

PERSON = (100, 100, 200, 400)                      # 100 × 300


def test_ioa():
    assert ioa((0, 0, 10, 10), (0, 0, 20, 20)) == 1.0
    assert ioa((0, 0, 10, 10), (5, 0, 20, 20)) == 0.5


def test_compliant_worker(policy, det):
    v = evaluate([det("Person", PERSON), det("helmet", (130, 100, 170, 140)), det("vest", (110, 170, 190, 280))],
                 policy)
    (p,) = v.people
    assert p.items_required == {"helmet": OK, "vest": OK}
    assert p.status == OK and v.status == OK and not v.unassigned


def test_negative_class_is_a_violation(policy, det):
    v = evaluate([det("Person", PERSON), det("no_helmet", (130, 100, 170, 140)), det("vest", (110, 170, 190, 280))],
                 policy)
    assert v.people[0].items_required["helmet"] == VIOLATION
    assert v.status == VIOLATION and v.violations == 1


def test_nothing_seen_is_unverified_unless_policy_says_otherwise(policy, det, tmp_path):
    v = evaluate([det("Person", PERSON), det("vest", (110, 170, 190, 280))], policy)
    assert v.people[0].items_required["helmet"] == UNVERIFIED and v.status == UNVERIFIED
    raw = yaml.safe_load(open("config/site_policy.yaml"))
    raw["missing_is_violation"] = True
    (tmp_path / "p.yaml").write_text(yaml.safe_dump(raw))
    strict = load_policy(tmp_path / "p.yaml")
    assert evaluate([det("Person", PERSON), det("vest", (110, 170, 190, 280))], strict).status == VIOLATION


def test_helmet_must_be_on_the_head(policy, det):
    # a helmet carried at hip height is inside the person box but not in the head region
    v = evaluate([det("Person", PERSON), det("helmet", (120, 260, 160, 300))], policy)
    assert v.people[0].items_required["helmet"] == UNVERIFIED
    assert [g.name for g in v.unassigned] == ["helmet"]


def test_gear_outside_any_person_is_unassigned(policy, det):
    v = evaluate([det("Person", PERSON), det("helmet", (400, 100, 440, 140))], policy)
    assert len(v.unassigned) == 1


def test_overlapping_workers_get_their_own_gear(policy, det):
    front = (150, 150, 230, 400)                    # nearer worker, smaller box inside the crowd
    back = (100, 100, 260, 420)
    dets = [det("Person", back), det("Person", front), det("helmet", (175, 150, 205, 180)),
            det("no_helmet", (160, 100, 200, 140))]
    v = evaluate(dets, policy)
    by_box = {p.box: p for p in v.people}
    assert by_box[tuple(map(float, front))].items_required["helmet"] == OK
    assert by_box[tuple(map(float, back))].items_required["helmet"] == VIOLATION


def test_conflicting_boxes_take_the_more_confident_one(policy, det):
    head = (130, 100, 170, 140)
    dets = [det("Person", PERSON), det("helmet", head, 0.4), det("no_helmet", head, 0.8)]
    assert evaluate(dets, policy).people[0].items_required["helmet"] == VIOLATION
    dets = [det("Person", PERSON), det("helmet", head, 0.9), det("no_helmet", head, 0.3)]
    assert evaluate(dets, policy).people[0].items_required["helmet"] == OK


def test_no_people_means_ok_frame(policy, det):
    v = evaluate([det("helmet", (0, 0, 10, 10))], policy)
    assert v.people == [] and v.status == OK and len(v.unassigned) == 1


def test_policy_validation(tmp_path):
    raw = yaml.safe_load(open("config/site_policy.yaml"))
    raw["required"].append("harness")
    (tmp_path / "p.yaml").write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="harness"):
        load_policy(tmp_path / "p.yaml")


def test_to_dict(policy, det):
    d = evaluate([det("Person", PERSON), det("no_vest", (110, 170, 190, 280))], policy).to_dict()
    assert d["status"] == VIOLATION and d["persons"][0]["required"] == {"helmet": UNVERIFIED, "vest": VIOLATION}
